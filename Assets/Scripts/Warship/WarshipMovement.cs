using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class WarshipMovement : MonoBehaviour
{
    public int m_PlayerNumber = 1;
    public float m_DragInWaterForward = 100f;
    public bool m_IsHumanPlayer = false;

    [Header("UI Components")]
    public Text m_VelocityTextField;
    public Text m_RudderTextField;

    [Header("Engine Power")]
    public float m_EnginePower = 0.25f;
    public float m_RudderPower = 0.1f;

    private Rigidbody m_Rigidbody;
    private int m_VelocityLevel = 0;
    private int m_MinVelocityLevel = -2;
    private int m_MaxVelocityLevel = 4;
    private int m_SteerLevel = 0;
    private int m_MinSteerLevel = -2;
    private int m_MaxSteerLevel = 2;

    private RandomAgent m_RandomAgent;

    private void Awake()
    {
        m_Rigidbody = GetComponent<Rigidbody>();

        m_RandomAgent = new RandomAgent();
    }

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        if (m_IsHumanPlayer)
        {
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

            m_VelocityTextField.text = $"Velocity: {m_VelocityLevel}";
            m_RudderTextField.text = $"Steer: {m_SteerLevel}";
        }
        else
        {
            int action = m_RandomAgent.GetAction();
            if (action == 0)
            {
                // pass
            }
            else if (action == 1)
            {
                Accelerate(Direction.up);
            }
            else if (action == 2)
            {
                Accelerate(Direction.down);
            }
            else if (action == 3)
            {
                Steer(Direction.left);
            }
            else if (action == 4)
            {
                Steer(Direction.right);
            }
            else if (action == 5)
            {
                // Fire
            }
        }
    }

    private void FixedUpdate()
    {
        // UnityEngine.Profiling.Profiler.BeginSample("WarshipMovement.FixedUpdate");

        m_Rigidbody.AddRelativeForce(Vector3.forward * m_EnginePower * m_VelocityLevel);
        transform.rotation = Quaternion.Euler(0, transform.rotation.eulerAngles.y + m_RudderPower * m_SteerLevel, 0);

        // UnityEngine.Profiling.Profiler.EndSample();
    }

    public void Accelerate(Direction direction)
    {
        if (direction == Direction.up)
        {
            m_VelocityLevel = Mathf.Min(m_VelocityLevel + 1, m_MaxVelocityLevel);
        }
        else if (direction == Direction.down)
        {
            m_VelocityLevel = Mathf.Max(m_VelocityLevel - 1, m_MinVelocityLevel);
        }
    }

    public void Steer(Direction direction)
    {
        if (direction == Direction.left)
        {
            m_SteerLevel = Mathf.Max(m_SteerLevel - 1, m_MinSteerLevel);
        }
        else if (direction == Direction.right)
        {
            m_SteerLevel = Mathf.Min(m_SteerLevel + 1, m_MaxSteerLevel);
        }
    }

    private void OnDisable()
    {
        m_VelocityLevel = 0;
        m_SteerLevel = 0;

        //m_VelocityTextField = 
    }
}
