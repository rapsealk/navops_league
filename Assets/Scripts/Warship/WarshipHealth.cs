using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class WarshipHealth : MonoBehaviour
{
    public Slider m_Slider;
    public Color m_FullHealthColor = Color.green;

    public const float StartingHealth = 100f;
    public const float DefaultDamage = 10f;

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

    private void OnEnable()
    {
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
        m_Slider.value = m_CurrentHealth;
        // m_Slider.value = Mathf.Lerp(m_Slider.value, m_CurrentHealth, Time.deltaTime);
    }

    private void OnDeath()
    {
        m_IsDestroyed = true;

        this.gameObject.SetActive(false);
    }
}
