using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Engine : MonoBehaviour
{
    //private float m_HorsePower = 1f;
    private Rigidbody m_Rigidbody;

    private Pathfinder Pathfinder;
    private float HorsePower = 30f;

    // Start is called before the first frame update
    void Start()
    {
        m_Rigidbody = GetComponent<Rigidbody>();

        m_Rigidbody.constraints = RigidbodyConstraints.FreezeRotationX | RigidbodyConstraints.FreezeRotationZ;

        Pathfinder = GetComponent<Pathfinder>();
        List<Node> path = Pathfinder.FindPath(transform.position, new Vector3(-94f, 0f, 220f));
        Queue<Vector3> pathPoints = new Queue<Vector3>();
        for (int i = 0; i < path.Count; i++)
        {
            pathPoints.Enqueue(path[i].WorldPosition);
        }
        StartCoroutine(NavigateTo(pathPoints));

        /*Queue<Vector3> pathPoints = new Queue<Vector3>();
        pathPoints.Enqueue(new Vector3(-20f, 0f, 10f));
        pathPoints.Enqueue(new Vector3(-40f, 0f, -10f));
        pathPoints.Enqueue(new Vector3(-60f, 0f, 10f));
        StartCoroutine(NavigateTo(pathPoints));*/
    }

    // Update is called once per frame
    void Update()
    {
        float vertical = Input.GetAxisRaw("Vertical");
        float horizontal = Input.GetAxisRaw("Horizontal");

        //float vertical = Random.Range(-1f, 1f);
        //float horizontal = Random.Range(-1f, 1f);

        Steer(horizontal);
        Combust(vertical);
    }

    public void Combust(float fuel = 1.0f)
    {
        m_Rigidbody.AddForce(transform.forward * fuel * HorsePower, ForceMode.Acceleration);
    }

    public void Steer(float rudder = 1.0f)
    {
        m_Rigidbody.transform.Rotate(Vector3.up, rudder * 0.1f);
    }

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

                if (dir.magnitude > 8f || Mathf.Abs(degree - y) < 10f)
                {
                    Combust(Mathf.Min(1.0f, dir.sqrMagnitude * 0.5f));
                }

                //Debug.Log($"NavigateTo: {y} -> {degree} ({degree - y})");

                yield return null;
            }
        }
    }

    public bool ArrivedAt(Vector3 target)
    {
        return (transform.position - target).magnitude <= 1f;
    }
}
