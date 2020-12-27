using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(Boid))]
public class BoidAvoidanceBehaviour : MonoBehaviour
{
    public float Radius;
    public float RepulsionForce;

    private Boid Boid;

    // Start is called before the first frame update
    void Start()
    {
        Boid = GetComponent<Boid>();
    }

    // Update is called once per frame
    void Update()
    {
        Boid[] boids = FindObjectsOfType<Boid>();
        Vector3 average = Vector3.zero;
        uint found = 0;

        for (uint i = 0; i < boids.Length; i++)
        {
            Boid boid = boids[i];
            if (boid == Boid)
            {
                continue;
            }

            Vector3 diff = boid.transform.position - this.transform.position;
            if (diff.magnitude < Radius)
            {
                average += boid.velocity;
                found += 1;
            }
        }

        if (found > 0)
        {
            average /= found;
            Boid.velocity -= Vector3.Lerp(Boid.velocity, average, Time.deltaTime) * RepulsionForce;
        }
    }
}
