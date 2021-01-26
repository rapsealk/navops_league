using UnityEngine;
using UnityEngine.UI;

public class GameManager : MonoBehaviour
{
    public Warship player1;
    public Warship player2;
    public Text player1PositionText;
    public Text player1RotationText;
    public Text player1HpText;
    public Text player2PositionText;
    public Text player2RotationText;
    public Text player2HpText;

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        Vector3 position1 = player1.transform.position;
        Vector3 rotation1 = player1.transform.rotation.eulerAngles;
        player1PositionText.text = string.Format("({0:F2}, {1:F2})", position1.x, position1.z);
        player1RotationText.text = string.Format("{0:F2}", rotation1.y);
        player1HpText.text = string.Format("{0:F2}", player1.currentHealth);
        Vector3 position2 = player2.transform.position;
        Vector3 rotation2 = player2.transform.rotation.eulerAngles;
        player2PositionText.text = string.Format("({0:F2}, {1:F2})", position2.x, position2.z);
        player2RotationText.text = string.Format("{0:F2}", rotation2.y);
        player2HpText.text = string.Format("{0:F2}", player2.currentHealth);
    }
}
