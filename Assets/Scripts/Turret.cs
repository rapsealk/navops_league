using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Turret : MonoBehaviour
{
    public GameObject m_Projectile;
    public GameObject m_BattleShip;
    //public GameObject m_DirectionIndicator;
    public Transform m_Muzzle;
    public ParticleSystem m_MuzzleFlash;

    //private float m_InitialRotation;
    //private float m_RotationLimit = 30f;
    // private float m_RotationDistanceLimit = (Mathf.Sqrt(3) + 1) / Mathf.Sqrt(2);   // theta = 30'
    //private bool m_IsLocked = false;
    private float m_CooldownTime = 6f;
    private float m_CurrentCooldownTime = 6f;
    private bool m_IsLoaded = true;

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
            m_IsLoaded = (m_CurrentCooldownTime >= m_CooldownTime);
        }

        /*
        Vector3 rotation = m_DirectionIndicator.transform.rotation.eulerAngles - m_BattleShip.transform.rotation.eulerAngles;
        //Vector3 rotation = Quaternion.FromToRotation(m_BattleShip.transform.rotation.eulerAngles, m_DirectionIndicator.transform.rotation.eulerAngles).eulerAngles;
        // Pitch
        rotation.x = Mathf.Min(0, rotation.x);
        // Yaw
        if (m_InitialRotation == 0)
        {
            // FIXME: x % 360
            if (rotation.y >= 360 - m_RotationLimit || rotation.y <= m_RotationLimit)
            {
                transform.Rotate(rotation - transform.rotation.eulerAngles, Space.Self);
                // transform.rotation = Quaternion.Euler(rotation);
                m_IsLocked = true;
            }
            //if (rotation.y >= 180 && rotation.y < 360 - m_RotationLimit)
            //{
            //    rotation.y = 360 - m_RotationLimit;
            //}
            //else if (rotation.y > m_RotationLimit)
            //{
            //    rotation.y = m_RotationLimit;
            //}
        }
        else if (m_InitialRotation == 90)
        {
            if (rotation.y >= 90 - m_RotationLimit && rotation.y <= 90 + m_RotationLimit)
            {
                transform.rotation = Quaternion.Euler(rotation);
                m_IsLocked = true;
            }
            //if (rotation.y > 90 + m_RotationLimit && rotation.y <= 270)
            //{
            //    rotation.y = 90 + m_RotationLimit;
            //}
            //else if (rotation.y > 270 || rotation.y < 90 - m_RotationLimit)
            //{
            //    rotation.y = 90 - m_RotationLimit;
            //}
        }
        else if (m_InitialRotation == 180)
        {
            if (rotation.y >= 180 - m_RotationLimit && rotation.y <= 180 + m_RotationLimit)
            {
                transform.rotation = Quaternion.Euler(rotation);
                m_IsLocked = true;
            }
            //if (rotation.y > 180 + m_RotationLimit && rotation.y <= 360)
            //{
            //    rotation.y = 180 + m_RotationLimit;
            //}
            //else if (rotation.y < 180 - m_RotationLimit)
            //{
            //    rotation.y = 180 - m_RotationLimit;
            //}
        }
        else if (m_InitialRotation == 270)
        {
            if (rotation.y >= 270 - m_RotationLimit && rotation.y <= 270 + m_RotationLimit)
            {
                transform.rotation = Quaternion.Euler(rotation);
                m_IsLocked = true;
            }
            //if (rotation.y > 270 + m_RotationLimit || rotation.y <= 90)
            //{
            //    rotation.y = 270 + m_RotationLimit;
            //}
            //else if (rotation.y < 270 - m_RotationLimit)
            //{
            //    rotation.y = 270 - m_RotationLimit;
            //}
        }
        */

        // transform.rotation = Quaternion.Euler(rotation);

        int layerMask = ~(1 << 8);
        RaycastHit hit;
        bool isLocked = Physics.Raycast(m_Muzzle.position, m_Muzzle.forward, out hit, Mathf.Infinity, layerMask);

        if (isLocked)
        {
            Debug.DrawRay(m_Muzzle.position, m_Muzzle.forward * hit.distance, Color.green);
        }
        else
        {
            Debug.DrawRay(m_Muzzle.position, m_Muzzle.forward * 1000, Color.red);
        }

        if (Input.GetKeyDown(KeyCode.Mouse0))
        {
            if (isLocked && m_IsLoaded)
            {
                GameObject bullet = Instantiate(m_Projectile, m_Muzzle.position, m_Muzzle.rotation);
                //bullet.GetComponent<Renderer>().material.SetColor("_Color", Color.red);
                bullet.GetComponent<Rigidbody>().AddForce(m_Muzzle.forward * 1000 + m_Muzzle.up * 10);
                Physics.IgnoreCollision(bullet.GetComponent<Collider>(), GetComponent<Collider>());
                m_MuzzleFlash.Play();

                //m_IsLoaded = false;
                m_CurrentCooldownTime = 0f;
            }
        }
    }
}
