using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Engine : MonoBehaviour
{
    //private float m_HorsePower = 1f;
    private Rigidbody m_Rigidbody;

    private float HorsePower = 30f;

    // Start is called before the first frame update
    void Start()
    {
        m_Rigidbody = GetComponent<Rigidbody>();

        m_Rigidbody.constraints = RigidbodyConstraints.FreezeRotationX | RigidbodyConstraints.FreezeRotationZ;
    }

    // Update is called once per frame
    void Update()
    {
        float vertical = Input.GetAxisRaw("Vertical");
        float horizontal = Input.GetAxisRaw("Horizontal");

        //float vertical = Random.Range(-1f, 1f);
        //float horizontal = Random.Range(-1f, 1f);

        Steer(horizontal);
        Combust(vertical);
    }

    public void Combust(float fuel = 1.0f)
    {
        m_Rigidbody.AddForce(transform.forward * fuel * HorsePower, ForceMode.Acceleration);
    }

    public void Steer(float rudder = 1.0f)
    {
        m_Rigidbody.transform.Rotate(Vector3.up, rudder * 0.1f);
    }

    public void NavigateTo(Vector3 target)
    {
        // NotImplemented
    }
}
