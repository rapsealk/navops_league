using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;

public class WarshipAcademyBehaviour : MonoBehaviour
{
    private void Awake()
    {
        Academy.Instance.OnEnvironmentReset += EnvironmentReset;
    }
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    private void EnvironmentReset()
    {
        Debug.Log("Academy.EnvironmentReset");
    }
}
