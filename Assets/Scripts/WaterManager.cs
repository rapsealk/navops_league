using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(MeshFilter))]
[RequireComponent(typeof(MeshRenderer))]
public class WaterManager : MonoBehaviour
{
    private MeshFilter m_MeshFilter;

    private void Awake()
    {
        m_MeshFilter = GetComponent<MeshFilter>();
    }

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        Vector3[] vertices = m_MeshFilter.mesh.vertices;
        for (int i = 0; i < vertices.Length; i++)
        {
            vertices[i].y = WaveManager.m_Instance.GetWaveHeight(transform.position.x + vertices[i].x);
        }

        m_MeshFilter.mesh.vertices = vertices;
        m_MeshFilter.mesh.RecalculateNormals();
    }
}
