using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;

public class Warship : Agent
{
    public const float m_Durability = 1000f;
    public Transform startingPoint;
    public Color rendererColor;
    public ParticleSystem explosion;
    public Warship target;
    [HideInInspector] public Rigidbody rb;
    public int playerId;
    public int teamId;
    [HideInInspector] public WeaponSystemsOfficer weaponSystemsOfficer;
    [HideInInspector] public float currentHealth { get; private set; }
    public Transform battleField;

    private Engine m_Engine;

    public void Reset()
    {
        GetComponent<Rigidbody>().velocity = Vector3.zero;
        GetComponent<Rigidbody>().angularVelocity = Vector3.zero;

        transform.position = startingPoint.position;
        transform.rotation = startingPoint.rotation;

        currentHealth = m_Durability;
    }

    // Start is called before the first frame update
    void Start()
    {
        weaponSystemsOfficer = GetComponent<WeaponSystemsOfficer>();
        weaponSystemsOfficer.Assign(teamId, playerId);

        rb = GetComponent<Rigidbody>();
        rb.constraints = RigidbodyConstraints.FreezeRotationX | RigidbodyConstraints.FreezeRotationZ;

        m_Engine = GetComponent<Engine>();

        MeshRenderer[] meshRenderers = GetComponentsInChildren<MeshRenderer>();
        for (int i = 0; i < meshRenderers.Length; i++)
        {
            meshRenderers[i].material.color = rendererColor;
        }

        Reset();
    }

    // Update is called once per frame
    void Update()
    {
        if (Application.platform == RuntimePlatform.WindowsEditor)
        {
            if (Input.GetKeyDown(KeyCode.Mouse0))
            {
                weaponSystemsOfficer.FireMainBattery();
            }
            else if (Input.GetKeyDown(KeyCode.Mouse1))
            {
                // TODO: Animation
                weaponSystemsOfficer.FireTorpedoAt(target.transform.position);
            }
        }

        KeepTrackOnTarget();
    }

    private void KeepTrackOnTarget()
    {
        Vector3 rotation = Vector3.zero;
        rotation.y = Geometry.GetAngleBetween(transform.position, target.transform.position);

        Vector3 projectilexz = transform.position;
        projectilexz.y = 0f;
        Vector3 targetxz = target.transform.position;
        targetxz.y = 0f;
        float r = Vector3.Distance(projectilexz, targetxz);
        float G = Physics.gravity.y;
        float vz = 8000f;
        rotation.x = Mathf.Atan((G * r) / (vz * 2f)) * Mathf.Rad2Deg;   // max: 140

        weaponSystemsOfficer.Aim(Quaternion.Euler(rotation));
    }

    #region MLAgent
    public override void Initialize()
    {
        // Academy.Instance.AutomaticSteppingEnabled = false;
    }

    public override void OnEpisodeBegin()
    {
        Reset();
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        // Player
        sensor.AddObservation(transform.position.x / (battleField.transform.localScale.x / 2) - 1f);
        sensor.AddObservation(transform.position.z / (battleField.transform.localScale.z / 2) - 1f);

        float radian = (transform.rotation.eulerAngles.y % 360) * Mathf.Deg2Rad;
        sensor.AddObservation(Mathf.Cos(radian));
        sensor.AddObservation(Mathf.Sin(radian));

        // Opponent
        sensor.AddObservation(target.transform.position.x / (battleField.transform.localScale.x / 2) - 1f);
        sensor.AddObservation(target.transform.position.z / (battleField.transform.localScale.z / 2) - 1f);

        float targetRadian = (target.transform.rotation.eulerAngles.y % 360) * Mathf.Deg2Rad;
        sensor.AddObservation(Mathf.Cos(targetRadian));
        sensor.AddObservation(Mathf.Sin(targetRadian));

        bool isEnemyTorpedoLaunched = false;
        Vector3 enemyTorpedoPosition = Vector3.zero;
        GameObject torpedo = target.weaponSystemsOfficer.torpedoInstance;
        if (torpedo != null)
        {
            isEnemyTorpedoLaunched = true;
            enemyTorpedoPosition = torpedo.transform.position;
        }
        sensor.AddObservation(isEnemyTorpedoLaunched);
        sensor.AddObservation(enemyTorpedoPosition.x / (battleField.transform.localScale.x / 2) - 1f);
        sensor.AddObservation(enemyTorpedoPosition.z / (battleField.transform.localScale.z / 2) - 1f);

        // Weapon
        WeaponSystemsOfficer.BatterySummary[] batterySummary = weaponSystemsOfficer.Summary();
        for (int i = 0; i < batterySummary.Length; i++)
        {
            WeaponSystemsOfficer.BatterySummary summary = batterySummary[i];
            sensor.AddObservation(Mathf.Cos(summary.rotation.x));
            sensor.AddObservation(Mathf.Sin(summary.rotation.x));
            sensor.AddObservation(Mathf.Cos(summary.rotation.y));
            sensor.AddObservation(Mathf.Sin(summary.rotation.y));
            sensor.AddObservation(summary.isReloaded);
            sensor.AddObservation(summary.cooldown);
            sensor.AddObservation(summary.isDamaged);
            sensor.AddObservation(summary.repairProgress);
        }
        sensor.AddObservation(weaponSystemsOfficer.isTorpedoReady);
        sensor.AddObservation(weaponSystemsOfficer.torpedoCooldown / WeaponSystemsOfficer.m_TorpedoReloadTime);

        // Penalty
        float distance = Vector3.Distance(transform.position, target.transform.position);
        if (distance >= 200f)
        {
            float penalty = -Mathf.Pow(distance - 200f, 2f) / 10000f;
            AddReward(penalty);
        }
        
        float fuelLoss = 1 / 10000f;
        AddReward(fuelLoss);
    }

    public override void OnActionReceived(float[] vectorAction)
    {
        // Interpret signals
        float enginePower = Mathf.Clamp(vectorAction[0], -1f, 1f);
        float rudderPower = Mathf.Clamp(vectorAction[1], -1f, 1f);
        bool fireMainBattery = (vectorAction[2] >= 0.5f);
        bool launchTorpedo = (vectorAction[3] >= 0.5f);

        m_Engine.Combust(enginePower);
        m_Engine.Steer(rudderPower);

        if (fireMainBattery)
        {
            weaponSystemsOfficer.FireMainBattery();
        }

        if (launchTorpedo)
        {
            weaponSystemsOfficer.FireTorpedoAt(target.transform.position);
        }
    }

    public override void Heuristic(float[] actionsOut)
    {
        //
    }
    #endregion  // MLAgent

    public void OnCollisionEnter(Collision collision)
    {
        if (collision.collider.name.StartsWith("Water"))
        {
            return;
        }

        explosion.transform.position = collision.transform.position;
        explosion.transform.rotation = collision.transform.rotation;
        explosion.Play();

        if (collision.collider.tag == "Player")
        {
            SetReward(-1.0f);
            target.SetReward(-1.0f);
            EndEpisode();
            return;
        }
        else if (collision.collider.tag == "Torpedo")
        {
            currentHealth = 0f;
        }
        else if (collision.collider.tag.StartsWith("Bullet")
                 && !collision.collider.tag.EndsWith(teamId.ToString()))
        {
            float damage = collision.rigidbody?.velocity.magnitude ?? 0f;
            Debug.Log($"[{teamId}-{playerId}] OnCollisionEnter(collision: {collision.collider.name}) ({collision.collider.tag} / {damage})");
            currentHealth -= damage;
        }
        else if (collision.collider.tag == "Terrain")
        {
            float damage = rb.velocity.magnitude * rb.mass;
            Debug.Log($"[{teamId}-{playerId}] OnCollisionEnter(collision: {collision.collider.name}) ({collision.collider.tag} / {damage})");
            currentHealth -= damage;
        }

        if (currentHealth <= 0f + Mathf.Epsilon)
        {
            SetReward(-10.0f);
            target.SetReward(10.0f);
            EndEpisode();
        }
    }

    public void OnTriggerEnter(Collider other)
    {
        explosion.transform.position = other.transform.position;
        explosion.transform.rotation = other.transform.rotation;
        explosion.Play();
    }
}
