using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class WaveManager : MonoBehaviour
{
    public static WaveManager m_Instance;

    public float m_Amplitude = 1f;
    public float m_Length = 2f;
    public float m_Speed = 1f;
    public float m_Offset = 0f;

    private void Awake()
    {
        if (m_Instance == null)
        {
            m_Instance = this;
        }
        else if (m_Instance != this)
        {
            Destroy(this);
        }
    }

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        m_Offset += Time.deltaTime * m_Speed;
    }

    public float GetWaveHeight(float x)
    {
        return m_Amplitude * Mathf.Sin(x / m_Length + m_Offset);
    }
}
