using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;

public class WarshipAgent : Agent
{
    public int m_PlayerId;
    public Color m_RendererColor;
    public Transform m_StartingPoint;
    public ParticleSystem m_ExplosionAnimation;
    public WarshipAgent m_Opponent;
    public DominationManager m_DominationManager;
    [Header("Maneuver Parameters")]
    public const float m_EnginePower = 8f;
    public const float m_RudderPower = 0.1f;

    public const float StartingHealth = 100f;
    public const float DefaultDamage = 10f;

    private Transform m_Transform;
    private Rigidbody m_Rigidbody;
    private float m_CurrentHealth;
    private Transform m_OpponentTransform;
    private Turret[] m_Turrets;

    private float m_OpponentHealth;

    // Velocity
    private int m_VelocityLevel = 0;
    private const int minVelocityLevel = -2;
    private const int maxVelocityLevel = 4;
    private int m_SteerLevel = 0;
    private const int minSteerLevel = -2;
    private const int maxSteerLevel = 2;

    private const float winReward = 1.0f;
    private const float damageReward = -0.01f;

    public override void Initialize()
    {
        m_Transform = GetComponent<Transform>();
        m_Rigidbody = GetComponent<Rigidbody>();

        m_Turrets = GetComponentsInChildren<Turret>();
        for (int i = 0; i < m_Turrets.Length; i++)
        {
            m_Turrets[i].m_PlayerNumber = m_PlayerId;
        }
        Debug.Log($"[WarshipAgent-{m_PlayerId}] Initialize: {m_Turrets.Length} turrets.");

        MeshRenderer[] renderers = GetComponentsInChildren<MeshRenderer>();
        for (int i = 0; i < renderers.Length; i++)
        {
            renderers[i].material.color = m_RendererColor;
        }

        // MaxStep = 1000;
    }

    public override void OnEpisodeBegin()
    {
        Reset();
        m_Opponent.Reset();

        m_DominationManager.Reset();
    }

    public void Reset()
    {
        m_Rigidbody.velocity = Vector3.zero;
        m_Rigidbody.angularVelocity = Vector3.zero;

        m_Transform.localPosition = m_StartingPoint.localPosition;
        m_Transform.rotation = m_StartingPoint.rotation;
        m_CurrentHealth = StartingHealth;

        m_OpponentTransform = m_Opponent.GetComponent<Transform>();
        m_OpponentHealth = m_Opponent.m_CurrentHealth;

        m_VelocityLevel = 0;
        m_SteerLevel = 0;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(m_Transform.localPosition / 50f);   // 3 (x, y, z)
        sensor.AddObservation(m_Transform.rotation);        // 3 (x, y, z)
        sensor.AddObservation(m_CurrentHealth / StartingHealth);             // 1
        //sensor.AddObservation(m_Health.m_IsDestroyed);      // 1

        for (int i = 0; i < m_Turrets.Length; i++)
        {
            sensor.AddObservation(m_Turrets[i].CurrentCooldownTime);    // 6
        }

        sensor.AddObservation(m_OpponentTransform.localPosition / 50f);
        sensor.AddObservation(m_OpponentTransform.rotation);
        sensor.AddObservation(m_Opponent.m_CurrentHealth / StartingHealth);
        //sensor.AddObservation(m_OpponentHealth.m_IsDestroyed);
        // ...
    }

    public override void OnActionReceived(float[] vectorAction)
    {
        for (int i = 0; i < vectorAction.Length; i++)
        {
            if (vectorAction[i] == 1.0f)
            {
                if (i == 0)
                {
                    // NOOP
                }
                else if (i == 1)
                {
                    // W
                    Accelerate(Direction.up);
                }
                else if (i == 2)
                {
                    // S
                    Accelerate(Direction.down);
                }
                else if (i == 3)
                {
                    // A
                    Steer(Direction.left);
                }
                else if (i == 4)
                {
                    // D
                    Steer(Direction.right);
                }
                else if (i == 5)
                {
                    // FIRE
                    Fire();
                }
            }
        }
        // ...

        // Reward
        if (m_OpponentHealth - m_Opponent.m_CurrentHealth > 0f)
        {
            AddReward((m_OpponentHealth - m_Opponent.m_CurrentHealth) * damageReward);
            m_OpponentHealth = m_Opponent.m_CurrentHealth;
        }

        if (m_PlayerId == 1 && m_DominationManager.IsBlueDominating)
        {
            AddReward(0.01f);

            if (m_DominationManager.IsDominated)
            {
                SetReward(winReward);
                EndEpisode();
                m_Opponent.SetReward(-winReward);
                m_Opponent.EndEpisode();
            }
        }
        else if (m_PlayerId == 2 && m_DominationManager.IsRedDominating)
        {
            AddReward(0.01f);

            if (m_DominationManager.IsDominated)
            {
                SetReward(winReward);
                EndEpisode();
                m_Opponent.SetReward(-winReward);
                m_Opponent.EndEpisode();
            }
        }

        if (m_Transform.position.y <= 0.0f)
        {
            Vector3 position = transform.position;
            position.y = 0f;
            transform.position = position;
            // TakeDamage(StartingHealth);
        }

        if (m_Opponent.m_CurrentHealth <= 0f)
        {
            if (m_CurrentHealth > 0f)
            {
                SetReward(winReward);
            }
            else
            {
                SetReward(-winReward);
            }
            EndEpisode();
            m_Opponent.SetReward(-winReward);
            m_Opponent.EndEpisode();
        }
    }

    public override void Heuristic(float[] actionsOut)
    {
        // ...
        //Accelerate(Direction.up);
        //if (m_PlayerId == 2)
        //Steer(Direction.right);
        //Fire();
        Debug.Log($"Heuristic - {actionsOut.Length}");

        if (Input.GetKeyDown(KeyCode.W))
        {
            Accelerate(Direction.up);
        }
        else if (Input.GetKeyDown(KeyCode.S))
        {
            Accelerate(Direction.down);
        }
        
        if (Input.GetKeyDown(KeyCode.A))
        {
            Steer(Direction.left);
        }
        else if (Input.GetKeyDown(KeyCode.D))
        {
            Steer(Direction.right);
        }

        if (Input.GetKeyDown(KeyCode.Mouse0))
        {
            Fire();
        }
    }

    private void FixedUpdate()
    {
        m_Rigidbody.AddRelativeForce(Vector3.forward * m_EnginePower * m_VelocityLevel, ForceMode.Acceleration);
        //m_Rigidbody.AddRelativeTorque(Vector3.up * m_RudderPower * m_SteerLevel, ForceMode.Force);
        transform.rotation = Quaternion.Euler(0, transform.rotation.eulerAngles.y + m_RudderPower * m_SteerLevel, 0);

        Debug.Log($"[Warship#{m_PlayerId}] transform.position.y: {transform.position.y}");
    }

    void OnCollisionEnter(Collision collision)
    {
        if (collision.collider.tag == "Wall")
        {
            TakeDamage(DefaultDamage);
        }
    }

    void OnTriggerEnter(Collider collider)
    {
        Debug.Log($"ID #{m_PlayerId} [WarshipHealth.OnTriggerEnter] {collider} {collider.tag}");
        m_ExplosionAnimation.Play();

        if (collider.CompareTag("Battleship"))
        {
            TakeDamage(WarshipHealth.StartingHealth);
        }
        else if (collider.CompareTag("Bullet") /*&& !collider.tag.EndsWith(m_PlayerId.ToString())*/)
        {
            TakeDamage(WarshipHealth.DefaultDamage);
        }
    }

    public void TakeDamage(float damage)
    {
        m_CurrentHealth -= damage;

        AddReward(damage * damageReward);
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

    public void Fire()
    {
        for (int j = 0; j < m_Turrets.Length; j++)
        {
            m_Turrets[j].Fire();
        }
    }
}
