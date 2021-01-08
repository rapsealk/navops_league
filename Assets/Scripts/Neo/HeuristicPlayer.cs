using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class HeuristicPlayer : MonoBehaviour
{
    private Engine engine;
    private float waitingTime = 3f;
    private float battleTime = 0f;
    //private Pathfinder m_Pathfinder;

    // Start is called before the first frame update
    void Start()
    {
        engine = GetComponent<Engine>();

        //m_Pathfinder = GetComponent<Pathfinder>();
    }

    // Update is called once per frame
    void Update()
    {
        battleTime += Time.deltaTime;
        if (battleTime >= waitingTime)
        {
            // engine.FindPathTo();
        }

        /* Human Controller
        Vector3 pointer = Input.mousePosition;
        Ray cast = Camera.main.ScreenPointToRay(pointer);
        int waterLayerMask = 1 << 4;
        if (Physics.Raycast(cast, out RaycastHit hit, Mathf.Infinity, waterLayerMask))
        {
            Debug.Log($"RaycastHit: {hit.point}");
            FireTorpedoAt(hit.point, cameraQuaternion.eulerAngles);
        }
        */
    }

    /*
    public IEnumerator NavigateTo(Queue<Vector3> pathPoints)
    {
        while (pathPoints.Count > 0)
        {
            Vector3 target = pathPoints.Dequeue();

            while (!ArrivedAt(target))
            {
                Vector3 dir = target - transform.position;
                Debug.DrawRay(transform.position, dir, Color.red);
                float degree = Geometry.GetAngleBetween(transform.position, target);
                degree = (degree + 360) % 360;
                degree = (degree > 180f) ? (degree - 360f) : degree;
                float y = (transform.rotation.eulerAngles.y + 360) % 360;
                y = (y > 180f) ? (y - 360f) : y;

                if (Mathf.Abs(degree - y) < 180f)
                {
                    Steer(Mathf.Sign(degree - y));
                }
                else
                {
                    Steer(Mathf.Sign(y - degree));
                }

                if (dir.magnitude > 10f || Mathf.Abs(degree - y) < 90f)
                {
                    Combust(Mathf.Min(1.0f, dir.magnitude * 0.5f));
                }

                //Debug.Log($"NavigateTo: {y} -> {degree} ({degree - y})");

                yield return null;
            }
        }
    }

    public void FindPathTo(Vector3 position)
    {
        Vector3 target = new Vector3(159.7f, 0f, 216.8f); // new Vector3(-94f, 0f, 220f)
        List<Node> path = Pathfinder.FindPath(transform.position, target);
        Queue<Vector3> pathPoints = new Queue<Vector3>();
        for (int i = 0; i < path.Count; i++)
        {
            pathPoints.Enqueue(path[i].WorldPosition);
        }
        StartCoroutine(NavigateTo(pathPoints));
    }

    public bool ArrivedAt(Vector3 target)
    {
        return (transform.position - target).magnitude <= 10f;
    }
    */

    /*
    public GameObject Opponent;
    public float RadarRange = 100f;

    public readonly IdleState<HeuristicPlayer> IdleState = new IdleState<HeuristicPlayer>();
    public readonly BattleState<HeuristicPlayer> BattleState = new BattleState<HeuristicPlayer>();
    private BaseState<HeuristicPlayer> CurrentState;

    public Engine Engine { get; private set; }
    public Pathfinder Pathfinder { get; private set; }

    // Start is called before the first frame update
    void Start()
    {
        Engine = GetComponent<Engine>();
        Pathfinder = GetComponent<Pathfinder>();

        Transition(IdleState);
    }

    // Update is called once per frame
    void Update()
    {
        CurrentState.Update(this);
    }

    public void Transition(BaseState<HeuristicPlayer> state)
    {
        CurrentState = state;
        CurrentState.EnterState(this);
    }
    */
}
