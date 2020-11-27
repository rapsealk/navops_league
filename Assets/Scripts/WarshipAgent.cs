using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;

public class WarshipAgent : Agent
{
    public int m_PlayerId;
    [HideInInspector]
    public Warship m_Warship;
    public Warship m_Opponent;
    public DominationManager m_DominationManager;

    public enum ActionId
    {
        NOOP = 0,
        FORWARD = 1,
        BACKWARD = 2,
        LEFT = 3,
        RIGHT = 4,
        FIRE = 5
    }

    [Header("Finite State Machine")]
    public bool m_IsFiniteStateMachineBot = false;
    // [HideInInspector]
    private WarshipControllerFSM m_FiniteStateMachine = null;

    private Transform m_OpponentTransform;

    private float m_OpponentHealth;

    private const float winReward = 1.0f;
    private const float damageReward = -0.01f;

    public override void Initialize()
    {
        m_Warship = GetComponent<Warship>();
        m_Warship.m_PlayerId = m_PlayerId;

        if (m_IsFiniteStateMachineBot)
        {
            m_FiniteStateMachine = new WarshipControllerFSM();
            m_FiniteStateMachine.m_Opponent = m_Opponent;
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
        m_Warship.Reset();

        m_OpponentTransform = m_Opponent.GetComponent<Transform>();
        m_OpponentHealth = m_Opponent.m_CurrentHealth;

        if (m_FiniteStateMachine != null)
        {
            m_FiniteStateMachine?.TransitionToState(m_FiniteStateMachine.m_IdleState);
        }
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(m_Warship.m_Transform.localPosition / 100f);          // 3 (x, y, z)
        sensor.AddObservation(m_Warship.m_Transform.rotation);                      // 3 (x, y, z)
        sensor.AddObservation(m_Warship.m_CurrentHealth / Warship.StartingHealth);  // 1

        for (int i = 0; i < m_Warship.m_Turrets.Length; i++)
        {
            sensor.AddObservation(m_Warship.m_Turrets[i].CurrentCooldownTime);      // 6
        }

        sensor.AddObservation(m_OpponentTransform.localPosition / 100f);
        sensor.AddObservation(m_OpponentTransform.rotation);
        sensor.AddObservation(m_Opponent.m_CurrentHealth / Warship.StartingHealth);

        // Reward
        #region RewardShaping

        if (m_OpponentHealth > m_Opponent.m_CurrentHealth)
        {
            AddReward((m_OpponentHealth - m_Opponent.m_CurrentHealth) * damageReward);
            m_OpponentHealth = m_Opponent.m_CurrentHealth;
        }

        /*
        if (m_PlayerId == 1 && m_DominationManager.IsBlueDominating)
        {
            AddReward(0.01f);

            if (m_DominationManager.IsDominated)
            {
                SetReward(winReward);
                EndEpisode();
                //m_Opponent.SetReward(-winReward);
                //m_Opponent.EndEpisode();
            }
        }
        else if (m_PlayerId == 2 && m_DominationManager.IsRedDominating)
        {
            AddReward(0.01f);

            if (m_DominationManager.IsDominated)
            {
                SetReward(winReward);
                EndEpisode();
                //m_Opponent.SetReward(-winReward);
                //m_Opponent.EndEpisode();
            }
        }
        */

        if (m_Warship.m_Transform.position.y <= 0.0f)
        {
            Vector3 position = transform.position;
            position.y = 0f;
            transform.position = position;
        }

        if (m_Opponent.m_CurrentHealth <= 0f)
        {
            SetReward(winReward);
            EndEpisode();
            //m_Opponent.SetReward(-winReward);
            //m_Opponent.EndEpisode();
        }
        else if (m_Warship.m_CurrentHealth <= 0f)
        {
            SetReward(-winReward);
            EndEpisode();
        }

        #endregion
    }

    public override void OnActionReceived(float[] vectorAction)
    {
        for (int i = 0; i < vectorAction.Length; i++)
        {
            if (vectorAction[i] == 1.0f)
            {
                if (i == (int) ActionId.NOOP)
                {
                    // NOOP
                }
                else if (i == (int) ActionId.FORWARD)
                {
                    m_Warship.Accelerate(Direction.up);
                }
                else if (i == (int) ActionId.BACKWARD)
                {
                    m_Warship.Accelerate(Direction.down);
                }
                else if (i == (int) ActionId.LEFT)
                {
                    m_Warship.Steer(Direction.left);
                }
                else if (i == (int) ActionId.RIGHT)
                {
                    m_Warship.Steer(Direction.right);
                }
                else if (i == (int) ActionId.FIRE)
                {
                    m_Warship.Fire();
                }
            }
        }
    }

    public override void Heuristic(float[] actionsOut)
    {
        // TODO: Bot
        if (Input.GetKeyDown(KeyCode.W))
        {
            actionsOut[(int)ActionId.FORWARD] = 1;
        }
        else if (Input.GetKeyDown(KeyCode.S))
        {
            actionsOut[(int)ActionId.BACKWARD] = 1;
        }
        else if (Input.GetKeyDown(KeyCode.A))
        {
            actionsOut[(int)ActionId.LEFT] = 1;
        }
        else if (Input.GetKeyDown(KeyCode.D))
        {
            actionsOut[(int)ActionId.RIGHT] = 1;
        }
        else if (Input.GetKeyDown(KeyCode.Mouse0))
        {
            actionsOut[(int)ActionId.FIRE] = 1;
        }
    }

    private void TakeDamage(float damage)
    {
        m_Warship.TakeDamage(damage);
        AddReward(damage * damageReward);
    }

    void OnCollisionEnter(Collision collision)
    {
        if (collision.collider.tag == "Wall")
        {
            TakeDamage(Warship.DefaultDamage);
        }
    }

    void OnTriggerEnter(Collider collider)
    {
        //Debug.Log($"ID #{m_PlayerId} [WarshipHealth.OnTriggerEnter] {collider} {collider.tag}");
        m_Warship.m_ExplosionAnimation.Play();

        /*
        if (collider.CompareTag("Battleship"))
        {
            TakeDamage(WarshipHealth.StartingHealth);
            Debug.Log($"ID#{m_PlayerId} - {collider.tag} -> {m_Warship.m_CurrentHealth}");
        }
        else*/
        if (collider.tag.Contains("Bullet") && !collider.tag.EndsWith(m_PlayerId.ToString()))
        {
            TakeDamage(WarshipHealth.DefaultDamage);
            Debug.Log($"ID#{m_PlayerId} - {collider.tag} -> {m_Warship.m_CurrentHealth}");
        }
    }
}
