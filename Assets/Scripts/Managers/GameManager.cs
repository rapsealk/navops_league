using System.Collections;
using System.Linq;
using UnityEngine;

public class GameManager : MonoBehaviour
{
    public float m_StartDelay = 0.1f;
    public float m_EndDelay = 0.1f;
    // public CameraControl m_CameraControl;
    public CameraController m_CameraController;
    public GameObject m_WarshipPrefab;
    public ParticleSystem m_ExplosionAnimation;
    public WarshipManager[] m_Warships;

    private WaitForSeconds m_StartWait;
    private WaitForSeconds m_EndWait;
    // private WarshipManager m_GameWinner;

    // Start is called before the first frame update
    void Start()
    {
        m_StartWait = new WaitForSeconds(m_StartDelay);
        m_EndWait = new WaitForSeconds(m_EndDelay);

        SpawnAllWarships();
        // SetCameraTargets();

        StartCoroutine(Loop());
    }

    /* Update is called once per frame
    void Update()
    {
        
    }
    */

    private void SpawnAllWarships()
    {
        Debug.Log("[GameManager] SpawnAllWarships");

        for (int i = 0; i < m_Warships.Length; i++)
        {
            m_Warships[i].m_Instance =
                Instantiate(m_WarshipPrefab, m_Warships[i].m_SpawnPoint.position, m_Warships[i].m_SpawnPoint.rotation) as GameObject;
            m_Warships[i].m_PlayerNumber = i + 1;
            //m_Warships[i].m_ExplosionAnimation = m_ExplosionAnimation;
            m_Warships[i].m_ExplosionAnimation = Instantiate<ParticleSystem>(m_ExplosionAnimation, m_Warships[i].m_Instance.transform);
            m_Warships[i].Setup();
        }
    }

    /*
    private void SetCameraTargets()
    {
        Transform[] targets = new Transform[m_Warships.Length];

        for (int i = 0; i < targets.Length; i++)
        {
            targets[i] = m_Warships[i].m_Instance.transform;
        }

        m_CameraControl.m_Targets = targets;
    }
    */

    private IEnumerator Loop()
    {
        yield return StartCoroutine(OnBattleReady());
        yield return StartCoroutine(OnBattleProgress());
        yield return StartCoroutine(OnBattleFinish());

        StartCoroutine(Loop());
    }

    private IEnumerator OnBattleReady()
    {
        ResetAllWarships();
        DisableWarshipControl();

        m_CameraController.m_Target = m_Warships.First(warship => warship.m_IsHumanPlayer).m_Instance.transform;
        //m_CameraControl.SetStartPositionAndSize();

        yield return m_StartWait;
    }

    private IEnumerator OnBattleProgress()
    {
        EnableWarshipControl();

        while (!IsGameFinished())
        {
            yield return null;
        }
    }

    private IEnumerator OnBattleFinish()
    {
        DisableWarshipControl();

        // TODO: m_GameWinner

        yield return m_EndWait;
    }

    private bool IsGameFinished()
    {
        return m_Warships.Any(warship => !warship.m_Instance.activeSelf);
    }

    private void ResetAllWarships()
    {
        Debug.Log("[GameManager] ResetAllWarships");

        for (int i = 0; i < m_Warships.Length; i++)
        {
            m_Warships[i].Reset();
        }
    }

    private void EnableWarshipControl()
    {
        for (int i = 0; i < m_Warships.Length; i++)
        {
            m_Warships[i].EnableControl();
        }
    }

    private void DisableWarshipControl()
    {
        for (int i = 0; i < m_Warships.Length; i++)
        {
            m_Warships[i].DisableControl();
        }
    }

    /*
    private IEnumerator OnBattleStart()
    {

    }
    */
}
