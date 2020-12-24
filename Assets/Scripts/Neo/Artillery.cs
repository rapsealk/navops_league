using UnityEngine;

public enum TurretType
{
    FRONTAL = 0,
    RIGHT = 1,
    REAR = 2,
    LEFT = 3
}

public class Artillery : MonoBehaviour
{
    public GameObject m_ShellPrefab;
    public Transform m_Muzzle;
    public ParticleSystem m_MuzzleFlash;
    [HideInInspector]
    public float Traverse = 90f;
    [HideInInspector]
    public float TraverseSpeed = 15f;

    private TurretType TurretType;
    private bool Locked = true;
    private float InitialEulerRotation;
    private Vector2 FirePower = new Vector2(8000f, 600f);

    // Start is called before the first frame update
    void Start()
    {
        InitialEulerRotation = (transform.localRotation.eulerAngles.y + 360) % 360;

        if (InitialEulerRotation <= Mathf.Epsilon)
        {
            TurretType = TurretType.FRONTAL;
        }
        else if (InitialEulerRotation <= 90f + Mathf.Epsilon)
        {
            TurretType = TurretType.RIGHT;
        }
        else if (InitialEulerRotation <= 180f + Mathf.Epsilon)
        {
            TurretType = TurretType.REAR;
        }
        else
        {
            TurretType = TurretType.LEFT;
        }

        // Debug.Log($"{GetType().Name}({name} {TurretType}) InitialEulerRotation: {InitialEulerRotation}");
    }

    // Update is called once per frame
    void Update()
    {
        /*
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
        */
    }

    public void Fire()
    {
        if (!Locked)
        {
            return;
        }

        m_MuzzleFlash.Play();

        GameObject projectile = Instantiate(m_ShellPrefab, m_Muzzle.position + m_Muzzle.forward * 3, m_Muzzle.rotation);
        projectile.GetComponent<Rigidbody>().AddForce(m_Muzzle.transform.forward * FirePower.x
                                                    + m_Muzzle.transform.up * FirePower.y);
    }

    public void Rotate(Quaternion target)
    {
        // Base: Horizontal, Barrel: Vertical
        bool locked = true;
        Vector3 rotation = target.eulerAngles;

        float x = (rotation.x + 360) % 360;
        if (x < 180f + 0f)
        {
            x = 0f;
        }
        else if (360 - x > 60f)
        {
            x = -60f;
        }
        rotation.x = x;
        rotation.y = (rotation.y + 360) % 360;

        //transform.rotation = Quaternion.Euler(rotation);
        transform.rotation = Quaternion.RotateTowards(transform.rotation, Quaternion.Euler(rotation), TraverseSpeed * Time.deltaTime);

        ///
        /// Post-processing
        ///
        Vector3 localRotation = transform.localRotation.eulerAngles;
        localRotation.y = (localRotation.y > 180f) ? (localRotation.y - 360f) : localRotation.y;
        switch (TurretType)
        {
            case TurretType.FRONTAL:
                if (Mathf.Abs(localRotation.y) >= Traverse + Mathf.Epsilon)
                {
                    localRotation.y = Mathf.Sign(localRotation.y) * Traverse;
                    transform.localRotation = Quaternion.Euler(localRotation);

                    locked = false;
                }
                break;
            case TurretType.REAR:
                if (Mathf.Abs(localRotation.y) <= 180f - (Traverse + Mathf.Epsilon))
                {
                    localRotation.y = 180f - Mathf.Sign(localRotation.y) * Traverse;
                    transform.localRotation = Quaternion.Euler(localRotation);

                    locked = false;
                }
                break;
            case TurretType.LEFT:
                if (Mathf.Abs(localRotation.y + 90f) >= Traverse + Mathf.Epsilon)
                {
                    localRotation.y = -90f + Mathf.Sign(localRotation.y + 90f) * Traverse;
                    transform.localRotation = Quaternion.Euler(localRotation);

                    locked = false;
                }
                break;
            case TurretType.RIGHT:
                if (Mathf.Abs(localRotation.y - 90f) >= Traverse + Mathf.Epsilon)
                {
                    localRotation.y = 90f + Mathf.Sign(localRotation.y - 90f) * Traverse;
                    transform.localRotation = Quaternion.Euler(localRotation);

                    locked = false;
                }
                break;
        }

        Locked = locked;
    }
}
