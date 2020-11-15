using UnityEngine;
using UnityEngine.UI;

[System.Serializable]
public class WarshipManager
{
    public Color m_PlayerColor;
    public Transform m_SpawnPoint;
    public Slider m_HealthSlider;
    public Text m_VelocityText;
    public Text m_RudderText;
    public Slider[] m_TurretCooldownIndicators;

    [HideInInspector] public int m_PlayerNumber;
    //[HideInInspector] public string m_ColoredPlayerText;
    [HideInInspector] public GameObject m_Instance;
    [HideInInspector] public ParticleSystem m_ExplosionAnimation;
    // [HideInInspector] public int m_Wins;
    public bool m_IsHumanPlayer;

    // private TankMovement m_Movement;
    // private TankShooting m_Shooting;
    // private GameObject m_CanvasGameObject;
    private WarshipMovement m_Movement;
    private WarshipHealth m_Health;

    public void Setup()
    {
        Debug.Log($"[WarshipManager#{m_PlayerNumber}] Setup");

        // m_Movement = m_Instance.GetComponent<TankMovement>();
        // m_Shooting = m_Instance.GetComponent<TankShooting>();
        // m_CanvasGameObject = m_Instance.GetComponentInChildren<Canvas>().gameObject;
        m_Movement = m_Instance.GetComponent<WarshipMovement>();
        m_Health = m_Instance.GetComponent<WarshipHealth>();

        m_Movement.m_PlayerNumber = m_PlayerNumber;
        m_Health.m_PlayerNumber = m_PlayerNumber;
        // Setup UI Components
        m_Movement.m_VelocityTextField = m_VelocityText;
        m_Movement.m_RudderTextField = m_RudderText;
        m_Movement.m_IsHumanPlayer = m_IsHumanPlayer;

        m_Health.m_Slider = m_HealthSlider;
        m_Health.m_ExplosionAnimation = m_ExplosionAnimation;
        // m_Shooting.m_PlayerNumber = m_PlayerNumber;

        MeshRenderer[] renderers = m_Instance.GetComponentsInChildren<MeshRenderer>();
        for (int i = 0; i < renderers.Length; i++)
        {
            renderers[i].material.color = m_PlayerColor;
        }

        Turret[] turrets = m_Movement.GetComponentsInChildren<Turret>();
        for (int i = 0; i < turrets.Length; i++)
        {
            turrets[i].m_PlayerNumber = m_PlayerNumber;
            // turrets[i].m_TurretId = i + 1;
            if (m_TurretCooldownIndicators.Length > i)
            {
                turrets[i].m_CooldownIndicator = m_TurretCooldownIndicators[i];
            }
        }
    }

    public void DisableControl()
    {
        m_Movement.enabled = false;
        // m_Shooting.enabled = false;

        // m_CanvasGameObject.SetActive(false);
    }

    public void EnableControl()
    {
        m_Movement.enabled = true;
        // m_Shooting.enabled = true;

        // m_CanvasGameObject.SetActive(true);
    }

    public void Reset()
    {
        m_Movement.GetComponent<Rigidbody>().velocity = Vector3.zero;
        m_Movement.GetComponent<Rigidbody>().angularVelocity = Vector3.zero;

        m_Instance.transform.position = m_SpawnPoint.position;
        m_Instance.transform.rotation = m_SpawnPoint.rotation;

        m_Instance.SetActive(false);
        m_Instance.SetActive(true);
    }
}
