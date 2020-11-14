using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class WarshipMovement : MonoBehaviour
{
    public int m_PlayerNumber = 1;
    public float m_Health = 1000f;
    public float m_DragInWaterForward = 100f;
    public ParticleSystem m_ExplosionAnimation;
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

    private void Awake()
    {
        m_Rigidbody = GetComponent<Rigidbody>();
    }

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.W))
        {
            m_VelocityLevel = Mathf.Min(m_VelocityLevel + 1, m_MaxVelocityLevel);
        }
        else if (Input.GetKeyDown(KeyCode.S))
        {
            m_VelocityLevel = Mathf.Max(m_VelocityLevel - 1, m_MinVelocityLevel);
        }

        if (Input.GetKeyDown(KeyCode.A))
        {
            m_SteerLevel = Mathf.Max(m_SteerLevel - 1, m_MinSteerLevel);
        }
        else if (Input.GetKeyDown(KeyCode.D))
        {
            m_SteerLevel = Mathf.Min(m_SteerLevel + 1, m_MaxSteerLevel);
        }

        if (m_IsHumanPlayer)
        {
            m_VelocityTextField.text = $"Velocity: {m_VelocityLevel}";
            m_RudderTextField.text = $"Steer: {m_SteerLevel}";
        }
    }

    private void FixedUpdate()
    {
        // UnityEngine.Profiling.Profiler.BeginSample("WarshipMovement.FixedUpdate");

        m_Rigidbody.AddRelativeForce(Vector3.forward * m_EnginePower * m_VelocityLevel);
        transform.rotation = Quaternion.Euler(0, transform.rotation.eulerAngles.y + m_RudderPower * m_SteerLevel, 0);

        // UnityEngine.Profiling.Profiler.EndSample();
    }

    void OnTriggerEnter(Collider collider)
    {
        Debug.Log($"[WarshipMovement.OnTriggerEnter] {collider}");
        //m_ExplosionAnimation.transform.position = new Vector3(Random.Range(-1f, 1f), Random.Range(-1f, 1f), Random.Range(-1f, 1f));
        m_ExplosionAnimation.Play();

        m_Health -= 100;
        Debug.Log($"[WarShip:{m_PlayerNumber}] Health: {m_Health}");
    }
}
