using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class WeaponSystemsOfficer : MonoBehaviour
{
    [HideInInspector] public int playerId;
    [HideInInspector] public int teamId;
    public GameObject torpedoPrefab;
    [HideInInspector] public GameObject torpedoInstance = null;
    [HideInInspector] public float m_TorpedoReloadTime = 40f;

    private Artillery[] m_Batteries;
    private float m_TorpedoCooldownTimer = 0f;
    private bool m_TorpedoReloaded = true;

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        if (!m_TorpedoReloaded)
        {
            m_TorpedoCooldownTimer += Time.deltaTime;

            if (m_TorpedoCooldownTimer >= m_TorpedoReloadTime)
            {
                m_TorpedoReloaded = true;
            }
        }

        if (torpedoInstance != null)
        {
            Debug.Log($"WSO: Torpedo: {torpedoInstance.transform.position}");
        }
    }

    public void Assign(int teamId, int playerId)
    {
        this.teamId = teamId;
        this.playerId = playerId;

        m_Batteries = GetComponentsInChildren<Artillery>();
        for (int i = 0; i < m_Batteries.Length; i++)
        {
            m_Batteries[i].playerId = playerId;
            m_Batteries[i].teamId = teamId;
        }
    }

    public void Aim(Quaternion target)
    {
        for (int i = 0; i < m_Batteries.Length; i++)
        {
            m_Batteries[i].Rotate(target);
        }
    }

    public void FireMainBattery()
    {
        for (int i = 0; i < m_Batteries.Length; i++)
        {
            m_Batteries[i].Fire();
        }
    }

    public void FireTorpedoAt(Vector3 position)
    {
        if (!m_TorpedoReloaded)
        {
            return;
        }

        Vector3 releasePoint = transform.position + (position - transform.position).normalized * 8f;
        releasePoint.y = 0f;

        float y = Geometry.GetAngleBetween(transform.position, position);
        Vector3 rotation = new Vector3(90f, y, 0f);

        torpedoInstance = Instantiate(torpedoPrefab, releasePoint, Quaternion.Euler(rotation));

        m_TorpedoReloaded = false;
    }
}
