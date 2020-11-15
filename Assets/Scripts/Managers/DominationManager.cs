using UnityEngine;
using UnityEngine.UI;

public class DominationManager : MonoBehaviour
{
    public Slider m_DominationSlider;
    public const float RequiredDominationTime = 10f;
    [HideInInspector] public WarshipManager[] m_Warships = null;

    private bool IsBlueDominating = false;
    private bool IsRedDominating = false;
    public bool IsDominated
    {
        get { return DominationTime >= RequiredDominationTime; }
    }
    private float DominationTime = 0f;
    private float ControlAreaScale = 16f;

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        if (m_Warships != null)
        {
            // FIXME: Trigger
            WarshipManager blueManager = m_Warships[0];
            float blueDistance = Mathf.Sqrt(Mathf.Pow(blueManager.m_Instance.transform.position.x, 2f)
                                            + Mathf.Pow(blueManager.m_Instance.transform.position.z, 2f));
            IsBlueDominating = (blueDistance < ControlAreaScale);

            WarshipManager redManager = m_Warships[1];
            float redDistance = Mathf.Sqrt(Mathf.Pow(redManager.m_Instance.transform.position.x, 2f)
                                            + Mathf.Pow(redManager.m_Instance.transform.position.z, 2f));
            IsRedDominating = (redDistance < ControlAreaScale);

            Debug.Log($"[DominationManager]" +
                $" Warship#{m_Warships[0].m_PlayerNumber}: {m_Warships[0].m_Instance.transform.position} ({IsBlueDominating}) /" +
                $" Warship#{m_Warships[1].m_PlayerNumber}: {m_Warships[1].m_Instance.transform.position} ({IsRedDominating})");
        }

        if (IsBlueDominating ^ IsRedDominating)
        {
            DominationTime += Time.deltaTime;

            if (DominationTime >= RequiredDominationTime)
            {
                // TODO
            }

            UpdateUI();
        }
        else
        {
            Reset();
        }
    }

    void FixedUpdate()
    {

    }

    public void Init(WarshipManager[] warships)
    {
        m_Warships = (WarshipManager[]) warships.Clone();
    }

    public void Reset()
    {
        DominationTime = 0f;

        UpdateUI();
    }

    private void UpdateUI()
    {
        m_DominationSlider.value = DominationTime / RequiredDominationTime;

        if (IsBlueDominating)
        {
            m_DominationSlider.GetComponentsInChildren<Image>()[1].color = Color.blue;
        }
        else if (IsRedDominating)
        {
            m_DominationSlider.GetComponentsInChildren<Image>()[1].color = Color.red;
        }
    }
}
