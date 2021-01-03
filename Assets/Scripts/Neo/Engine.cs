using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Engine : MonoBehaviour
{
    private Rigidbody m_Rigidbody;

    private Pathfinder Pathfinder;
    private float HorsePower = 30f;

    // Start is called before the first frame update
    void Start()
    {
        m_Rigidbody = GetComponent<Rigidbody>();

        m_Rigidbody.constraints = RigidbodyConstraints.FreezeRotationX | RigidbodyConstraints.FreezeRotationZ;

        Pathfinder = GetComponent<Pathfinder>();
    }

    // Update is called once per frame
    void Update()
    {
        float vertical = Input.GetAxisRaw("Vertical");
        float horizontal = Input.GetAxisRaw("Horizontal");

        Steer(horizontal * 30f);
        Combust(vertical * 30f);
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

                if (/*dir.magnitude > 10f || */Mathf.Abs(degree - y) < 90f)
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
}
