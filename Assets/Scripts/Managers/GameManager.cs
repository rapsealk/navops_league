using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GameManager : MonoBehaviour
{
    public float m_StartDelay = 3f;

    private WaitForSeconds m_StartWait;

    // Start is called before the first frame update
    void Start()
    {
        m_StartWait = new WaitForSeconds(m_StartDelay);
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    private IEnumerator Loop()
    {
        yield return StartCoroutine(OnBattleReady());
    }

    private IEnumerator OnBattleReady()
    {
        yield return m_StartWait;
    }

    /*
    private IEnumerator OnBattleStart()
    {

    }
    */
}
