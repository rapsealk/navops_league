using UnityEngine;
using Deprecated;

public class Warship : MonoBehaviour
{
    public int m_PlayerId;
    [HideInInspector]
    public Transform m_Transform;
    [HideInInspector]
    public Rigidbody m_Rigidbody;
    public Transform m_StartingPoint;
    public Color m_RendererColor;
    public ParticleSystem m_ExplosionAnimation = null;

    [Header("Maneuver Parameters")]
    public const float m_EnginePower = 2f;
    public const float m_RudderPower = 0.1f;

    [Header("Combat Parameters")]
    public const float StartingHealth = 100f;
    public const float DefaultDamage = 10f;
    [HideInInspector]
    public float m_CurrentHealth;
    [HideInInspector]
    public Turret[] m_Turrets;

    // Velocity
    private int m_VelocityLevel = 0;
    private const int minVelocityLevel = -2;
    private const int maxVelocityLevel = 4;
    private int m_SteerLevel = 0;
    private const int minSteerLevel = -2;
    private const int maxSteerLevel = 2;

    public enum EngineLevel
    {
        FORWARD_MAX = 4,
        FORWARD_3by4 = 3,
        FORWARD_HALF = 2,
        FORWARD_1by4 = 1,
        STOP = 0,
        BACKWARD_HALF = -1,
        BACKWARD_MAX = -2
    }

    public enum RudderLevel
    {
        LEFT_MAX = -2,
        LEFT_HALF = -1,
        STOP = 0,
        RIGHT_HALF = 1,
        RIGHT_MAX = 2
    }

    private void Awake()
    {
        m_Transform = GetComponent<Transform>();
        m_Rigidbody = GetComponent<Rigidbody>();

        MeshRenderer[] renderers = GetComponentsInChildren<MeshRenderer>();
        for (int i = 0; i < renderers.Length; i++)
        {
            renderers[i].material.color = m_RendererColor;
        }
    }

    // Start is called before the first frame update
    void Start()
    {
        m_Turrets = GetComponentsInChildren<Deprecated.Turret>();
        for (int i = 0; i < m_Turrets.Length; i++)
        {
            m_Turrets[i].m_PlayerNumber = m_PlayerId;
            m_Turrets[i].m_TurretId = i;
        }

        Reset();
    }

    // Update is called once per frame
    void Update()
    {
        Vector3 position = transform.position;

        if (transform.position.y <= 0.0f)
        {
            position.y = 0f;
        }

        if (Mathf.Abs(transform.position.x) > 90f)
        {
            position.x = Mathf.Sign(position.x) * 90f;
        }

        if (Mathf.Abs(transform.position.z) > 90f)
        {
            position.z = Mathf.Sign(position.z) * 90f;
        }

        transform.position = position;

        // Fire();
    }

    private void FixedUpdate()
    {
        m_Rigidbody.AddRelativeForce(Vector3.forward * m_EnginePower * m_VelocityLevel, ForceMode.Acceleration);
        transform.rotation = Quaternion.Euler(0, transform.rotation.eulerAngles.y + m_RudderPower * m_SteerLevel, 0);
    }

    /*
    void OnTriggerEnter(Collider collider)
    {
        m_ExplosionAnimation?.Play();
    }
    */

    public void Reset()
    {
        m_Rigidbody.velocity = Vector3.zero;
        m_Rigidbody.angularVelocity = Vector3.zero;

        m_Transform.localPosition = m_StartingPoint.localPosition;
        m_Transform.rotation = m_StartingPoint.rotation;
        m_CurrentHealth = StartingHealth;

        m_VelocityLevel = 0;
        m_SteerLevel = 0;
    }

    public void TakeDamage(float damage)
    {
        m_CurrentHealth -= damage;
    }

    public void Accelerate(Direction direction)
    {
        if (direction == Direction.up)
        {
            m_VelocityLevel = Mathf.Min(m_VelocityLevel + 1, maxVelocityLevel);
        }
        else if (direction == Direction.down)
        {
            m_VelocityLevel = Mathf.Max(m_VelocityLevel - 1, minVelocityLevel);
        }
    }

    public void Steer(Direction direction)
    {
        if (direction == Direction.left)
        {
            m_SteerLevel = Mathf.Max(m_SteerLevel - 1, minSteerLevel);
        }
        else if (direction == Direction.right)
        {
            m_SteerLevel = Mathf.Min(m_SteerLevel + 1, maxSteerLevel);
        }
    }

    public void SetEngineLevel(EngineLevel level)
    {
        m_VelocityLevel = (int) level;
    }

    public void SetRudderLevel(RudderLevel level)
    {
        m_SteerLevel = (int) level;
    }

    public void Fire()
    {
        for (int j = 0; j < m_Turrets.Length; j++)
        {
            m_Turrets[j].Fire();
        }
    }
}
