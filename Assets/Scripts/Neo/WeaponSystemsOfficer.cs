using UnityEngine;

public class WeaponSystemsOfficer : MonoBehaviour
{
    [HideInInspector] public int playerId;
    [HideInInspector] public int teamId;
    public GameObject torpedoPrefab;
    [HideInInspector] public GameObject torpedoInstance = null;
    [HideInInspector] public const float m_TorpedoReloadTime = 40f;
    [HideInInspector] public bool isTorpedoReady { get; private set; } = true;
    [HideInInspector] public float torpedoCooldown { get; private set; } = 0f;

    private Artillery[] m_Batteries;

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        if (!isTorpedoReady)
        {
            torpedoCooldown += Time.deltaTime;

            if (torpedoCooldown >= m_TorpedoReloadTime)
            {
                isTorpedoReady = true;
            }
        }

        if (torpedoInstance != null)
        {
            Debug.Log($"WSO: Torpedo: {torpedoInstance.transform.position}");
        }
    }

    public void Assign(int teamId, int playerId)
    {
        this.teamId = teamId;
        this.playerId = playerId;

        m_Batteries = GetComponentsInChildren<Artillery>();
        for (int i = 0; i < m_Batteries.Length; i++)
        {
            m_Batteries[i].playerId = playerId;
            m_Batteries[i].teamId = teamId;
        }
    }

    public void Aim(Quaternion target)
    {
        for (int i = 0; i < m_Batteries.Length; i++)
        {
            m_Batteries[i].Rotate(target);
        }
    }

    public void FireMainBattery()
    {
        for (int i = 0; i < m_Batteries.Length; i++)
        {
            m_Batteries[i].Fire();
        }
    }

    public void FireTorpedoAt(Vector3 position)
    {
        if (!isTorpedoReady)
        {
            return;
        }

        Vector3 releasePoint = transform.position + (position - transform.position).normalized * 8f;
        releasePoint.y = 0f;

        float y = Geometry.GetAngleBetween(transform.position, position);
        Vector3 rotation = new Vector3(90f, y, 0f);

        torpedoInstance = Instantiate(torpedoPrefab, releasePoint, Quaternion.Euler(rotation));

        isTorpedoReady = false;
    }

    public class BatterySummary
    {
        public Vector2 rotation;
        public bool isReloaded;
        public float cooldown;
        public bool isDamaged;
        public float repairProgress;

        public void Copy(Artillery battery)
        {
            Vector3 batteryRotation = battery.transform.rotation.eulerAngles;
            rotation = new Vector2(batteryRotation.x, batteryRotation.y);
            isReloaded = battery.isReloaded;
            cooldown = battery.cooldownTimer / Artillery.m_ReloadTime;
            isDamaged = battery.isDamaged;
            repairProgress = battery.repairTimer / Artillery.m_RepairTime;
        }
    }

    public BatterySummary[] Summary()
    {
        BatterySummary[] summary = new BatterySummary[m_Batteries.Length];
        for (int i = 0; i < m_Batteries.Length; i++)
        {
            summary[i] = new BatterySummary();
            summary[i].Copy(m_Batteries[i]);
        }
        return summary;
    }
}
