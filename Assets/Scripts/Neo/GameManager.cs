using UnityEngine;
using UnityEngine.UI;

public class GameManager : MonoBehaviour
{
    public Warship player1;
    public Warship player2;
    public Text player1Text;
    public Text player2Text;

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        Vector3 position1 = player1.transform.position;
        player1Text.text = string.Format("Player: ({0:F2}, {1:F2}) / {2:F}", position1.x, position1.z, player1.currentHealth);
        Vector3 position2 = player2.transform.position;
        player2Text.text = string.Format("Opponent: ({0:F2}, {1:F2}) / {2:F}", position2.x, position2.z, player2.currentHealth);
    }
}
