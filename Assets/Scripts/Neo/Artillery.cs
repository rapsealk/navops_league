using UnityEngine;

public enum TurretType
{
    FRONTAL = 0,
    RIGHT = 1,
    REAR = 2,
    LEFT = 3
}

public class Artillery : MonoBehaviour
{
    public GameObject shellPrefab;
    public Transform muzzle;
    public ParticleSystem muzzleFlash;
    [HideInInspector] public int playerId;
    [HideInInspector] public int teamId;
    [HideInInspector] public const float m_Traverse = 90f;
    [HideInInspector] public const float m_TraverseSpeed = 15f;
    [HideInInspector] public const float m_ReloadTime = 6f;
    [HideInInspector] public const float m_RepairTime = 30f;

    private TurretType m_TurretType;
    private float m_InitialEulerRotation;
    private Vector2 m_FirePower = new Vector2(8000f, 100f);
    private float m_CooldownTimer = 0f;
    private bool m_Reloaded = true;
    private float m_RepairTimer = 0f;
    private bool m_Damaged = false;

    // Start is called before the first frame update
    void Start()
    {
        m_InitialEulerRotation = (transform.localRotation.eulerAngles.y + 360) % 360;

        if (m_InitialEulerRotation <= Mathf.Epsilon)
        {
            m_TurretType = TurretType.FRONTAL;
        }
        else if (m_InitialEulerRotation <= 90f + Mathf.Epsilon)
        {
            m_TurretType = TurretType.RIGHT;
        }
        else if (m_InitialEulerRotation <= 180f + Mathf.Epsilon)
        {
            m_TurretType = TurretType.REAR;
        }
        else
        {
            m_TurretType = TurretType.LEFT;
        }

        // Debug.Log($"{GetType().Name}({name} {TurretType}) InitialEulerRotation: {InitialEulerRotation}");
    }

    // Update is called once per frame
    void Update()
    {
        if (m_Damaged)
        {
            m_RepairTimer += Time.deltaTime;

            if (m_RepairTimer >= m_RepairTime)
            {
                m_Damaged = false;
            }
        }
        else if (!m_Reloaded)
        {
            m_CooldownTimer += Time.deltaTime;

            if (m_CooldownTimer >= m_ReloadTime)
            {
                m_Reloaded = true;
            }
        }
    }

    public void Fire()
    {
        if (!m_Reloaded || m_Damaged)
        {
            return;
        }

        muzzleFlash.Play();

        GameObject projectile = Instantiate(shellPrefab, muzzle.position + muzzle.forward * 3, muzzle.rotation);

        Vector3 velocity = muzzle.transform.forward * m_FirePower.x + muzzle.transform.up * m_FirePower.y;
        Rigidbody rigidbody = projectile.GetComponent<Rigidbody>();
        rigidbody.velocity = velocity / rigidbody.mass;
        /*
        projectile.GetComponent<Rigidbody>().AddForce(muzzle.transform.forward * m_FirePower.x
                                                    + muzzle.transform.up * m_FirePower.y);
        */

        m_Reloaded = false;
        m_CooldownTimer = 0f;
    }

    public void Rotate(Quaternion target)
    {
        // TODO: Lock
        // Base: Horizontal, Barrel: Vertical
        Vector3 rotation = target.eulerAngles;

        float x = (rotation.x + 360) % 360;
        if (x < 180f)
        {
            x = 0f;
        }
        else if (360 - x > 60f)
        {
            x = -60f;
        }
        rotation.x = x;
        rotation.y = (rotation.y + 360) % 360;

        transform.rotation = Quaternion.RotateTowards(transform.rotation, Quaternion.Euler(rotation), m_TraverseSpeed * Time.deltaTime);

        ///
        /// Post-processing
        ///
        Vector3 localRotation = transform.localRotation.eulerAngles;
        localRotation.y = (localRotation.y > 180f) ? (localRotation.y - 360f) : localRotation.y;
        switch (m_TurretType)
        {
            case TurretType.FRONTAL:
                if (Mathf.Abs(localRotation.y) >= m_Traverse + Mathf.Epsilon)
                {
                    localRotation.y = Mathf.Sign(localRotation.y) * m_Traverse;
                    transform.localRotation = Quaternion.Euler(localRotation);
                }
                break;
            case TurretType.REAR:
                if (Mathf.Abs(localRotation.y) <= 180f - (m_Traverse + Mathf.Epsilon))
                {
                    localRotation.y = 180f - Mathf.Sign(localRotation.y) * m_Traverse;
                    transform.localRotation = Quaternion.Euler(localRotation);
                }
                break;
            case TurretType.LEFT:
                if (Mathf.Abs(localRotation.y + 90f) >= m_Traverse + Mathf.Epsilon)
                {
                    localRotation.y = -90f + Mathf.Sign(localRotation.y + 90f) * m_Traverse;
                    transform.localRotation = Quaternion.Euler(localRotation);
                }
                break;
            case TurretType.RIGHT:
                if (Mathf.Abs(localRotation.y - 90f) >= m_Traverse + Mathf.Epsilon)
                {
                    localRotation.y = 90f + Mathf.Sign(localRotation.y - 90f) * m_Traverse;
                    transform.localRotation = Quaternion.Euler(localRotation);
                }
                break;
        }
    }

    private void OnCollisionEnter(Collision collision)
    {
        // Debug.Log($"Artillery({name}).OnCollisionEnter(collision: {collision})");
    }

    private void OnTriggerEnter(Collider other)
    {
        // Debug.Log($"Artillery({name}).OnTriggerEnter(other: {other})");
        m_Damaged = true;
        m_RepairTimer = 0f;
    }
}
