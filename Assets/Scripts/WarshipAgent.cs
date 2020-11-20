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

    public const float StartingHealth = 100f;
    public const float DefaultDamage = 10f;

    private Transform m_Transform;
    private Rigidbody m_Rigidbody;
    private float m_CurrentHealth;
    private Transform m_OpponentTransform;

    // Velocity
    private int m_VelocityLevel = 0;
    private const int minVelocityLevel = -2;
    private const int maxVelocityLevel = 4;
    private int m_SteerLevel = 0;
    private const int minSteerLevel = -2;
    private const int maxSteerLevel = 2;

    private const float m_EnginePower = 0.25f;
    private const float m_RudderPower = 0.1f;

    private const float winReward = 1.0f;
    private const float damageReward = -0.01f;

    public override void Initialize()
    {
        m_Transform = GetComponent<Transform>();
        m_Rigidbody = GetComponent<Rigidbody>();

        MeshRenderer[] renderers = GetComponentsInChildren<MeshRenderer>();
        for (int i = 0; i < renderers.Length; i++)
        {
            renderers[i].material.color = m_RendererColor;
        }

        // MaxStep = 1000;
    }

    public override void OnEpisodeBegin()
    {
        m_Rigidbody.velocity = Vector3.zero;
        m_Rigidbody.angularVelocity = Vector3.zero;

        m_Transform.localPosition = m_StartingPoint.localPosition;
        m_Transform.rotation = m_StartingPoint.rotation;
        m_CurrentHealth = StartingHealth;

        m_OpponentTransform = m_Opponent.GetComponent<Transform>();

        m_VelocityLevel = 0;
        m_SteerLevel = 0;

        // StartCoroutine(ResetMaterial());
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(m_Transform.localPosition);   // 3 (x, y, z)
        sensor.AddObservation(m_Transform.rotation);        // 3 (x, y, z)
        sensor.AddObservation(m_CurrentHealth);             // 1
        //sensor.AddObservation(m_Health.m_IsDestroyed);      // 1

        sensor.AddObservation(m_OpponentTransform.localPosition);
        sensor.AddObservation(m_OpponentTransform.rotation);
        sensor.AddObservation(m_Opponent.m_CurrentHealth);
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
                }
            }
        }
        // ...

        // Reward
        if (m_Opponent.m_CurrentHealth <= 0f && m_CurrentHealth > 0f)
        {
            SetReward(winReward);
        }
        else
        {
            SetReward(0f);
        }
    }

    public override void Heuristic(float[] actionsOut)
    {
        // ...
    }

    private void FixedUpdate()
    {
        m_Rigidbody.AddRelativeForce(Vector3.forward * m_EnginePower * m_VelocityLevel);
        transform.rotation = Quaternion.Euler(0, transform.rotation.eulerAngles.y + m_RudderPower * m_SteerLevel, 0);
    }

    /*
    void OnCollisionEnter(Collision collision)
    {
        // SetReward(-1.0f);
        // EndEpisode();
    }
    */

    void OnTriggerEnter(Collider collider)
    {
        Debug.Log($"ID #{m_PlayerId} [WarshipHealth.OnTriggerEnter] {collider} {collider.tag}");
        m_ExplosionAnimation.Play();

        if (collider.tag == "Battleship")
        {
            TakeDamage(WarshipHealth.StartingHealth);
        }
        else if (collider.tag.StartsWith("Bullet") /*&& !collider.tag.EndsWith(m_PlayerId.ToString())*/)
        {
            TakeDamage(WarshipHealth.DefaultDamage);
        }
    }

    public void TakeDamage(float damage)
    {
        m_CurrentHealth -= damage;

        SetReward(damage * damageReward);

        if (m_CurrentHealth <= 0f)
        {
            EndEpisode();
        }
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
}
