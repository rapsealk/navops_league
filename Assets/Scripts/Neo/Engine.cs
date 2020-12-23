using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Engine : MonoBehaviour
{
    //private float m_HorsePower = 1f;
    private Rigidbody m_Rigidbody;

    // Start is called before the first frame update
    void Start()
    {
        m_Rigidbody = GetComponent<Rigidbody>();

        m_Rigidbody.constraints = RigidbodyConstraints.FreezeRotationX | RigidbodyConstraints.FreezeRotationZ;
    }

    // Update is called once per frame
    void Update()
    {
        float vertical = Input.GetAxisRaw("Vertical") * 30;
        float horizontal = Input.GetAxisRaw("Horizontal") * 1e-1f;

        m_Rigidbody.transform.Rotate(Vector3.up, horizontal);
        m_Rigidbody.AddForce(transform.forward * vertical, ForceMode.Acceleration);
        //m_Rigidbody.AddForce(transform.forward * m_HorsePower, ForceMode.Force);

        //Debug.Log($"[{GetType().Name}] Velocity: {m_Rigidbody.velocity.magnitude}");
    }
}
