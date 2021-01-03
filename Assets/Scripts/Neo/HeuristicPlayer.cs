using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class HeuristicPlayer : MonoBehaviour
{
    private Engine engine;
    private float waitingTime = 3f;
    private float battleTime = 0f;

    // Start is called before the first frame update
    void Start()
    {
        engine = GetComponent<Engine>();
    }

    // Update is called once per frame
    void Update()
    {
        battleTime += Time.deltaTime;
        if (battleTime >= waitingTime)
        {
            // engine.FindPathTo();
        }
    }
}
