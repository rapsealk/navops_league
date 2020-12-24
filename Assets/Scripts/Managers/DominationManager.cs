using UnityEngine;
using UnityEngine.UI;

/*
public class DominationManager : MonoBehaviour
{
    public bool m_IsEnabled = false;

    //public Slider m_DominationSlider;
    public const float RequiredDominationTime = 4f;
    public WarshipAgent m_BlueWarship;
    public WarshipAgent m_RedWarship;

    public bool IsBlueDominating { get => m_IsBlueDominating && !m_IsRedDominating; }
    public bool IsRedDominating { get => m_IsRedDominating && !m_IsBlueDominating; }
    private bool m_IsBlueDominating = false;
    private bool m_IsRedDominating = false;
    public bool IsDominated
    {
        get { return m_DominationTime >= RequiredDominationTime; }
    }
    private float m_DominationTime = 0f;
    public float DiminationTime { get => m_DominationTime; }
    private float ControlAreaScale = 25f;

    // Start is called before the first frame update
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {
        if (!m_IsEnabled)
        {
            return;
        }

        // FIXME: Trigger
        float blueDistance = Mathf.Sqrt(Mathf.Pow(m_BlueWarship.transform.position.x, 2f)
                                        + Mathf.Pow(m_BlueWarship.transform.position.z, 2f));
        m_IsBlueDominating = (blueDistance < ControlAreaScale);

        float redDistance = Mathf.Sqrt(Mathf.Pow(m_RedWarship.transform.position.x, 2f)
                                        + Mathf.Pow(m_RedWarship.transform.position.z, 2f));
        m_IsRedDominating = (redDistance < ControlAreaScale);

        //Debug.Log($"[DominationManager]" +
        //    $" Warship#{m_BlueWarship.m_PlayerId}: {m_BlueWarship.transform.position} ({m_IsBlueDominating}) /" +
        //    $" Warship#{m_RedWarship.m_PlayerId}: {m_RedWarship.transform.position} ({m_IsRedDominating})");

        if (m_IsBlueDominating ^ m_IsRedDominating)
        {
            m_DominationTime += Time.deltaTime;

            if (m_DominationTime >= RequiredDominationTime)
            {
                // TODO
            }

            //UpdateUI();
        }
        //else
        //{
        //    Reset();
        //}
    }

    void FixedUpdate()
    {

    }

    //public void Init(WarshipManager[] warships)
    //{
    //    m_Warships = (WarshipManager[]) warships.Clone();
    //}

    public void Reset()
    {
        m_DominationTime = 0f;

        // UpdateUI();
    }

    //private void UpdateUI()
    //{
    //    m_DominationSlider.value = DominationTime / RequiredDominationTime;

    //    if (IsBlueDominating)
    //    {
    //        m_DominationSlider.GetComponentsInChildren<Image>()[1].color = Color.blue;
    //    }
    //    else if (IsRedDominating)
    //    {
    //        m_DominationSlider.GetComponentsInChildren<Image>()[1].color = Color.red;
    //    }
    //}
}
*/