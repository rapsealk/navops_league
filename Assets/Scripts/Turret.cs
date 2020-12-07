using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class Turret : MonoBehaviour
{
    [HideInInspector] public int m_PlayerNumber;
    // [HideInInspector] public int m_TurretId;
    [HideInInspector] public Slider m_CooldownIndicator;
    public GameObject m_Projectile;
    //public GameObject m_DirectionIndicator;
    public Transform m_Muzzle;
    public ParticleSystem m_MuzzleFlash;

    private WarshipAgent m_WarshipAgent;
    private float m_RotationSpeed = 15f;
    private float m_RotationMaximum = 60f;
    private float m_RotationOffset = 0f;
    //private bool m_IsLocked = false;
    public const float reloadTime = 8f;
    private float m_CurrentCooldownTime = 6f;
    public float CurrentCooldownTime { get => Mathf.Min(reloadTime, m_CurrentCooldownTime) / reloadTime; }
    private bool m_IsLoaded = false;

    // Start is called before the first frame update
    void Start()
    {
        m_WarshipAgent = GetComponentInParent<WarshipAgent>();

        m_RotationOffset = GetComponent<Transform>().rotation.eulerAngles.y;

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

        Vector3 rotation = m_WarshipAgent.m_Opponent.m_Transform.rotation.eulerAngles - m_WarshipAgent.transform.rotation.eulerAngles;
        float rotation_y = Geometry.GetAngleBetween(m_WarshipAgent.transform.position, m_WarshipAgent.m_Opponent.m_Transform.position);
        if (rotation_y < 0)
        {
            rotation_y = 360 + rotation_y;
        }
        //Debug.Log($"rotation.y: {rotation.y} / rotation_y: {rotation_y}");
        // Pitch
        rotation.x = Mathf.Min(0, rotation.x);
        // Yaw
        if (m_RotationOffset == 0)
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
        }
        else if (m_RotationOffset == 90)
        {
            if (rotation_y >= 90 - m_RotationMaximum && rotation_y <= 90 + m_RotationMaximum)
            {
                //transform.rotation = Quaternion.Euler(rotation);
                //m_IsLocked = true;
            }
            if (rotation_y > 90 + m_RotationMaximum && rotation_y <= 270)
            {
                rotation_y = 90 + m_RotationMaximum;
            }
            else if (rotation_y > 270 || rotation_y < 90 - m_RotationMaximum)
            {
                rotation_y = 90 - m_RotationMaximum;
            }
        }
        else if (m_RotationOffset == 180)
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
        }
        else if (m_RotationOffset == 270)
        {
            if (rotation_y >= 270 - m_RotationMaximum && rotation_y <= 270 + m_RotationMaximum)
            {
                //transform.rotation = Quaternion.Euler(rotation);
                //m_IsLocked = true;
            }
            if (rotation_y > 270 + m_RotationMaximum || rotation_y <= 90)
            {
                rotation_y = 270 + m_RotationMaximum;
            }
            else if (rotation_y < 270 - m_RotationMaximum)
            {
                rotation_y = 270 - m_RotationMaximum;
            }
        }

        rotation.y = rotation_y;
        //rotation.y = Mathf.Lerp(rotation.y, rotation_y, Time.deltaTime);
        transform.rotation = Quaternion.Euler(rotation);
    }

    public void Fire()
    {
        if (!m_IsLoaded)
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
