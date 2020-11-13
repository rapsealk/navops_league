using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class WarshipMovement : MonoBehaviour
{
    public int m_PlayerNumber = 1;
    public float m_Health = 1000f;
    //public Text m_VelocityTextField;
    public float m_DragInWaterForward = 100f;
    public ParticleSystem m_ExplosionAnimation;

    [SerializeField] float m_Thrust;
    [SerializeField] float m_TurningSpeed;

    [Header("Engine Power")]
    float m_EnginePower = 0.1f;

    private Rigidbody m_Rigidbody;
    private int m_VelocityLevel = 0;
    private int m_MaxVelocityLevel = 4;
    private int m_MinVelocityLevel = -2;
    private float[] m_TargetVelocity = { -0.1f, -0.05f, 0f, 0.05f, 0.1f, 0.15f, 0.2f };
    private float m_CurrentVelocity = 0f;
    private const float m_VelocityAcceleration = 0.005f;

    private void Awake()
    {
        m_Rigidbody = GetComponent<Rigidbody>();
        //m_ExplosionAnimation = GetComponent<ParticleSystem>();
    }

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        /*
        if (Input.GetKeyDown(KeyCode.W))
        {
            m_VelocityLevel = Mathf.Min(m_VelocityLevel + 1, m_MaxVelocityLevel);
        }
        else if (Input.GetKeyDown(KeyCode.S))
        {
            m_VelocityLevel = Mathf.Max(m_VelocityLevel - 1, m_MinVelocityLevel);
        }

        m_VelocityTextField.text = $"Velocity: {m_VelocityLevel} ({m_CurrentVelocity} / {m_TargetVelocity[m_VelocityLevel+2]})";

        Accelerate();
        Move();
        */
    }

    private void FixedUpdate()
    {
        if (Input.GetKey(KeyCode.W))
        {
            m_Rigidbody.AddRelativeForce(Vector3.forward);
        }
        else if (Input.GetKey(KeyCode.S))
        {
            m_Rigidbody.AddRelativeForce(Vector3.back);
        }

        if (Input.GetKey(KeyCode.A))
        {
            transform.rotation = Quaternion.Euler(0, transform.rotation.eulerAngles.y - 0.2f, 0);
        }
        else if (Input.GetKey(KeyCode.D))
        {
            transform.rotation = Quaternion.Euler(0, transform.rotation.eulerAngles.y + 0.2f, 0);
        }

        /*
        // Move();
        UnityEngine.Profiling.Profiler.BeginSample("WarshipMovement.FixedUpdate");

        var position = transform.position;

        var forcePosition = m_Rigidbody.position;
        m_Rigidbody.AddForceAtPosition(transform.forward * Vector3.Dot(transform.forward, -m_Rigidbody.velocity.normalized) * m_DragInWaterForward, forcePosition, ForceMode.Acceleration);

        float forward = Input.GetAxis("Vertical");
        m_Rigidbody.AddForceAtPosition(transform.forward * m_EnginePower * forward, forcePosition, ForceMode.Acceleration);

        UnityEngine.Profiling.Profiler.EndSample();
        */
    }

    private void Move()
    {
        //transform.position += transform.rotation.normalized.eulerAngles * m_CurrentVelocity * Time.deltaTime;
        m_Rigidbody.AddForce(transform.rotation.normalized.eulerAngles * m_CurrentVelocity * 1000 * Time.deltaTime);
    }

    private void Accelerate()
    {
        float targetVelocity = m_TargetVelocity[m_VelocityLevel + 2];

        /*
        if (m_CurrentVelocity < targetVelocity)
        {
            m_CurrentVelocity += m_VelocityAcceleration;
        }
        else if (m_CurrentVelocity > targetVelocity)
        {
            m_CurrentVelocity -= m_VelocityAcceleration;
        }
        */

        m_CurrentVelocity = targetVelocity;
    }

    void OnTriggerEnter(Collider collider)
    {
        Debug.Log($"[WarshipMovement.OnTriggerEnter] {collider}");
        //m_ExplosionAnimation.transform.position = new Vector3(Random.Range(-1f, 1f), Random.Range(-1f, 1f), Random.Range(-1f, 1f));
        m_ExplosionAnimation.Play();

        m_Health -= 100;
        Debug.Log($"Health: {m_Health}");
    }
}
