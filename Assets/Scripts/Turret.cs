using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class Turret : MonoBehaviour
{
    [HideInInspector] public int m_PlayerNumber;
    [HideInInspector] public int m_TurretId;
    [HideInInspector] public Slider m_CooldownIndicator;
    public GameObject m_Projectile;
    //public GameObject m_DirectionIndicator;
    public Transform m_Muzzle;
    public ParticleSystem m_MuzzleFlash;

    private IWarshipController m_WarshipAgent;
    private float m_RotationSpeed = 15f;
    private float m_RotationMaximum = 60f;
    //private float m_RotationOffset = 0f;
    public const float reloadTime = 8f;
    public const float repairTime = 60f;
    private float m_CurrentCooldownTime = 6f;
    public float CurrentCooldownTime { get => Mathf.Min(reloadTime, m_CurrentCooldownTime) / reloadTime; }
    private bool m_IsLoaded = false;
    public bool m_IsDamaged = false;
    public float RepairTimeLeft = 0f;

    public enum TurretId
    {
        RIGHT_FRONTAL = 0,
        RIGHT_BACKWARD = 1,
        LEFT_FRONTAL = 2,
        LEFT_BACKWARD = 3,
        FRONTAL = 4,
        BACKWARD = 5
    }

    void Awake()
    {
        m_WarshipAgent = GetComponentInParent<IWarshipController>();

        //m_RotationOffset = GetComponent<Transform>().rotation.eulerAngles.y;
        //Debug.Log($"Start::RotationOffset: {m_RotationOffset}");
    }

    // Start is called before the first frame update
    void Start()
    {
        //m_InitialRotation = transform.rotation.eulerAngles.y;
        m_MuzzleFlash = m_Muzzle.GetComponentInChildren<ParticleSystem>();
        m_MuzzleFlash.transform.rotation = transform.rotation;
    }

    // Update is called once per frame
    void Update()
    {
        //m_IsLocked = false;

        if (!m_IsLoaded)
        {
            m_CurrentCooldownTime += Time.deltaTime;
            // UpdateUI();
            m_IsLoaded = (m_CurrentCooldownTime >= reloadTime);
        }

        if (m_IsDamaged)
        {
            RepairTimeLeft -= Time.deltaTime;
            if (RepairTimeLeft <= 0f)
            {
                m_IsDamaged = false;
            }
        }

        Vector3 rotation = m_WarshipAgent.GetOpponent().m_Transform.rotation.eulerAngles - m_WarshipAgent.GetTransform().rotation.eulerAngles;
        float rotation_y = Geometry.GetAngleBetween(m_WarshipAgent.GetTransform().position, m_WarshipAgent.GetOpponent().m_Transform.position) + transform.parent.rotation.eulerAngles.y;
        if (rotation_y < 0)
        {
            rotation_y = 360 + rotation_y;
        }
        //Debug.Log($"Id: {m_PlayerNumber} ({m_RotationOffset}) rotation.y: {rotation.y} / rotation_y: {rotation_y}");
        // Pitch
        rotation.x = Mathf.Min(0, rotation.x);
        // Yaw
        if (//m_RotationOffset == 0f
            m_TurretId == (int) TurretId.FRONTAL)
        {
            // FIXME: x % 360
            if (rotation_y >= 360 - m_RotationMaximum || rotation_y <= m_RotationMaximum)
            {
                //transform.Rotate(rotation - transform.rotation.eulerAngles, Space.Self);
                //rotation_y -= transform.rotation.y;
                // transform.rotation = Quaternion.Euler(rotation);
                //m_IsLocked = true;
            }
            else if (rotation_y >= 180 && rotation_y < 360 - m_RotationMaximum)
            {
                rotation_y = 360 - m_RotationMaximum;
            }
            else if (rotation_y > m_RotationMaximum)
            {
                rotation_y = m_RotationMaximum;
            }

            //rotation.y = rotation_y;
            //transform.localRotation = Quaternion.Euler(rotation);
        }
        else if (//m_RotationOffset == 180f
                m_TurretId == (int)TurretId.BACKWARD)
        {
            if (rotation_y >= 180 - m_RotationMaximum && rotation_y <= 180 + m_RotationMaximum)
            {
                //transform.Rotate(rotation - transform.rotation.eulerAngles, Space.Self);
                //rotation_y -= transform.rotation.y;
                //transform.rotation = Quaternion.Euler(rotation);
                //m_IsLocked = true;
            }
            else if (rotation_y > 180 + m_RotationMaximum && rotation_y <= 360)
            {
                rotation_y = 180 + m_RotationMaximum;
            }
            else if (rotation_y < 180 - m_RotationMaximum)
            {
                rotation_y = 180 - m_RotationMaximum;
            }

            //rotation.y = rotation_y;
            //transform.localRotation = Quaternion.Euler(rotation);
        }
        // ================================================
        else if (//m_RotationOffset == 90f
                m_TurretId == (int) TurretId.RIGHT_FRONTAL || m_TurretId == (int) TurretId.RIGHT_BACKWARD)
        {
            if (rotation_y >= 90 - m_RotationMaximum && rotation_y <= 90 + m_RotationMaximum)
            {
                //transform.rotation = Quaternion.Euler(rotation);
                //m_IsLocked = true;
            }
            else if (rotation_y > 90 + m_RotationMaximum && rotation_y <= 270)
            {
                rotation_y = 90 + m_RotationMaximum;
            }
            else if (rotation_y > 270 || rotation_y < 90 - m_RotationMaximum)
            {
                rotation_y = 90 - m_RotationMaximum;
            }
        }
        else if (//m_RotationOffset == 270f
                m_TurretId == (int) TurretId.LEFT_FRONTAL || m_TurretId == (int) TurretId.LEFT_BACKWARD)
        {
            if (rotation_y >= 270 - m_RotationMaximum && rotation_y <= 270 + m_RotationMaximum)
            {
                //transform.rotation = Quaternion.Euler(rotation);
                //m_IsLocked = true;
            }
            else if (rotation_y > 270 + m_RotationMaximum || rotation_y <= 90)
            {
                rotation_y = 270 + m_RotationMaximum;
            }
            else if (rotation_y < 270 - m_RotationMaximum)
            {
                rotation_y = 270 - m_RotationMaximum;
            }
        }

        //rotation.y = rotation_y;
        //rotation_y = Mathf.Sign(rotation_y) * Mathf.Min(Mathf.Abs(rotation_y), m_RotationSpeed);
        rotation.y = Mathf.LerpAngle(rotation.y, rotation_y, Mathf.Abs(rotation.y - rotation_y) / m_RotationSpeed);
        //rotation.y = rotation_y;
        transform.localRotation = Quaternion.Euler(rotation);

        /*
        if (Mathf.Abs(transform.localRotation.y) > m_RotationMaximum)
        {
            Vector3 localRotation = transform.localRotation.eulerAngles;
            localRotation.y = Mathf.Sign(localRotation.y) * m_RotationMaximum;
            transform.localRotation = Quaternion.Euler(localRotation);
        }*/

        /*
        int layerMask = 1 << 8;
        RaycastHit hit;
        if (!Physics.Raycast(m_Muzzle.position, m_Muzzle.forward, out hit, Mathf.Infinity, layerMask))
        {
            return;
        }*/

        Debug.DrawRay(m_Muzzle.position, m_Muzzle.forward, Color.green);
    }

    void OnTriggerEnter(Collider collider)
    {
        if (collider.tag.Contains("Bullet") && !collider.tag.EndsWith(m_PlayerNumber.ToString()))
        {
            //Debug.Log($"Turret#{m_PlayerNumber}-{m_TurretId} => TriggerEnter! ({collider.tag})");
            if (!m_IsDamaged)
            {
                RepairTimeLeft = repairTime;
                m_IsDamaged = true;
            }
        }
    }

    public void Fire()
    {
        if (!m_IsLoaded || m_IsDamaged)
        {
            return;
        }

        int layerMask = 1 << 8;
        RaycastHit hit;
        if (!Physics.Raycast(m_Muzzle.position, m_Muzzle.forward, out hit, Mathf.Infinity, layerMask))
        {
            return;
        }

        GameObject bullet = Instantiate(m_Projectile, m_Muzzle.position + m_Muzzle.forward * 3, m_Muzzle.rotation);
        bullet.tag = "Bullet" + m_PlayerNumber.ToString();
        //bullet.GetComponent<Renderer>().material.SetColor("_Color", Color.red);
        //bullet.GetComponent<Rigidbody>().AddForce(m_Muzzle.forward * 4000 + m_Muzzle.up * 10);
        bullet.GetComponent<Rigidbody>().AddForce(m_Muzzle.forward * 6000 + m_Muzzle.up * 20);
        //Physics.IgnoreCollision(bullet.GetComponent<Collider>(), GetComponent<Collider>());
        m_MuzzleFlash.Play();

        m_IsLoaded = false;
        m_CurrentCooldownTime = 0f;
    }

    /*
    private void UpdateUI()
    {
        var indicator = m_CooldownIndicator;
        if (indicator == null)
        {
            return;
        }

        indicator.value = m_CurrentCooldownTime / reloadTime;
    }
    */
}
