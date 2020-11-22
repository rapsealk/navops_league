using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Floater : MonoBehaviour
{
    public float m_DepthBeforeSubmerged = 1.0f;
    public float m_DisplacementAmount = 3.0f;
    public int m_FloaterCount = 1;
    public float m_WaterDrag = 0.99f;
    public float m_WaterAngularDrag = 0.5f;

    private Rigidbody m_Rigidbody;

    // Start is called before the first frame update
    void Start()
    {
        m_Rigidbody = GetComponent<Rigidbody>();
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    private void FixedUpdate()
    {
        m_Rigidbody.AddForceAtPosition(Physics.gravity / m_FloaterCount, transform.position, ForceMode.Acceleration);

        float waveHeight = WaveManager.m_Instance.GetWaveHeight(transform.position.x);

        if (transform.position.y < waveHeight)
        {
            float displacementMultiplier = Mathf.Clamp01((waveHeight - transform.position.y) / m_DepthBeforeSubmerged) * m_DisplacementAmount;
            // m_Rigidbody.AddForce(Vector3.up * Mathf.Abs(Physics.gravity.y) * displacementMultiplier, ForceMode.Acceleration);
            m_Rigidbody.AddForceAtPosition(Vector3.up * Mathf.Abs(Physics.gravity.y) * displacementMultiplier, transform.position, ForceMode.Acceleration);
            m_Rigidbody.AddForce(m_DisplacementAmount * -m_Rigidbody.velocity * m_WaterDrag * Time.fixedDeltaTime, ForceMode.VelocityChange);
            //m_Rigidbody.AddTorque(m_DisplacementAmount * -m_Rigidbody.angularVelocity * m_WaterAngularDrag * Time.fixedDeltaTime, ForceMode.VelocityChange);
        }
    }
}
