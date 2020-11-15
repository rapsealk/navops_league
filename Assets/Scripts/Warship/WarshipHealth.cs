using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class WarshipHealth : MonoBehaviour
{
    public Slider m_Slider;
    public Color m_FullHealthColor = Color.green;
    public ParticleSystem m_ExplosionAnimation;

    public const float StartingHealth = 100f;
    public const float DefaultDamage = 10f;

    [HideInInspector]
    public int m_PlayerNumber;

    private float m_CurrentHealth;
    private bool m_IsDestroyed;

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    void OnTriggerEnter(Collider collider)
    {
        Debug.Log($"ID #{m_PlayerNumber} [WarshipHealth.OnTriggerEnter] {collider} {collider.tag}");
        //m_ExplosionAnimation.transform.position = new Vector3(Random.Range(-1f, 1f), Random.Range(-1f, 1f), Random.Range(-1f, 1f));
        m_ExplosionAnimation.Play();

        //m_Health -= 100;
        //Debug.Log($"[WarShip:{m_PlayerNumber}] Health: {m_Health}");
        if (collider.tag == "Battleship")
        {
            TakeDamage(WarshipHealth.StartingHealth);
        }
        else if (collider.tag.StartsWith("Bullet") && !collider.tag.EndsWith(m_PlayerNumber.ToString()))
        {
            TakeDamage(WarshipHealth.DefaultDamage);
        }
    }

    private void OnEnable()
    {
        Debug.Log($"[WarshipHealth#{m_PlayerNumber}] OnEnable");

        m_CurrentHealth = StartingHealth;
        m_IsDestroyed = false;

        SetHealthUI();
    }

    public void TakeDamage(float damage)
    {
        m_CurrentHealth -= damage;

        SetHealthUI();

        if (m_CurrentHealth <= 0f && !m_IsDestroyed)
        {
            OnDeath();
        }
    }

    private void SetHealthUI()
    {
        if (m_Slider != null)
        {
            m_Slider.value = m_CurrentHealth;
        }
        // m_Slider.value = Mathf.Lerp(m_Slider.value, m_CurrentHealth, Time.deltaTime);
    }

    private void OnDeath()
    {
        m_IsDestroyed = true;

        this.gameObject.SetActive(false);
    }
}
