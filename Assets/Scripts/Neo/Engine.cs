using UnityEngine;

public class Engine : MonoBehaviour
{
    private Rigidbody m_Rigidbody;

    private float HorsePower = 30f;

    // Start is called before the first frame update
    void Start()
    {
        m_Rigidbody = GetComponent<Rigidbody>();
    }

    // Update is called once per frame
    void Update()
    {
        if (Application.platform == RuntimePlatform.WindowsEditor)
        {
            float vertical = Input.GetAxisRaw("Vertical");
            float horizontal = Input.GetAxisRaw("Horizontal");

            Steer(horizontal);
            Combust(vertical);
        }
    }

    public void Combust(float fuel = 1.0f)
    {
        m_Rigidbody.AddForce(transform.forward * fuel * HorsePower, ForceMode.Acceleration);
    }

    public void Steer(float rudder = 1.0f)
    {
        m_Rigidbody.transform.Rotate(Vector3.up, rudder * 0.1f);
    }
}
