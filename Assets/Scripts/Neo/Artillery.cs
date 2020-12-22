using UnityEngine;

public class Artillery : MonoBehaviour
{
    public GameObject m_ShellPrefab;
    public Transform m_Muzzle;
    public ParticleSystem m_MuzzleFlash;

    private Vector2 m_FirePower = new Vector2(8000f, 600f);

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Mouse0))
        {
            Debug.Log($"{GetType().Name}:Mouse0");
            Fire();
        }

        float horizontal = Input.GetAxis("Horizontal");
        float vertical = -1 * Input.GetAxis("Vertical");

        transform.Rotate(Vector3.up, horizontal);

        m_Muzzle.Rotate(Vector3.right, vertical);

        if (Mathf.Cos(m_Muzzle.transform.rotation.eulerAngles.x * Mathf.Deg2Rad) <= Mathf.Cos(15f * Mathf.Deg2Rad))
        {
            Vector3 rotation = m_Muzzle.transform.rotation.eulerAngles;
            Debug.Log($"{GetType().Name} rotation.x: {rotation.x} / Mathf.Sign: {Mathf.Sign(rotation.x)}");
            float sign = (rotation.x <= 180f) ? 1.0f : -1.0f;
            rotation.x = 15f * sign;
            m_Muzzle.transform.rotation = Quaternion.Euler(rotation);
        }
    }

    public void Fire()
    {
        m_MuzzleFlash.Play();

        GameObject projectile = Instantiate(m_ShellPrefab, m_Muzzle.position + m_Muzzle.forward * 3, m_Muzzle.rotation);
        projectile.GetComponent<Rigidbody>().AddForce(m_Muzzle.transform.forward * m_FirePower.x
                                                    + m_Muzzle.transform.up * m_FirePower.y);
    }
}
