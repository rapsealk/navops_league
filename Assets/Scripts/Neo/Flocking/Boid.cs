using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Boid : MonoBehaviour
{
    public Vector3 velocity;
    public float maxVelocity;

    // Start is called before the first frame update
    void Start()
    {
        velocity = new Vector3(Random.Range(-1f, -Mathf.Epsilon) * 5f, 0, Random.Range(Mathf.Epsilon, 1f) * 5f);
        maxVelocity = 20f;
    }

    // Update is called once per frame
    void Update()
    {
        if (velocity.magnitude > maxVelocity)
        {
            velocity = velocity.normalized * maxVelocity;
        }
        else if (velocity.magnitude <= Mathf.Epsilon)
        {
            velocity *= 5f;
            // velocity = new Vector3(Random.Range(-1f, -Mathf.Epsilon) * 5f, 0, Random.Range(Mathf.Epsilon, 1f) * 5f);
        }

        transform.position += velocity * 0.1f * Time.deltaTime;
        if (velocity.magnitude > Mathf.Epsilon)
        {
            transform.rotation = Quaternion.LookRotation(velocity);
        }
    }
}
