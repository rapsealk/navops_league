using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

[RequireComponent(typeof(Boid))]
public class BoidAlignmentBehaviour : MonoBehaviour
{
    public float Radius;

    private Boid Boid;

    // Start is called before the first frame update
    void Start()
    {
        Boid = GetComponent<Boid>();
    }

    // Update is called once per frame
    void Update()
    {
        /*
        Boid[] boids = FindObjectsOfType<Boid>();
        Vector3 average = Vector3.zero;
        uint found = 0;

        foreach (var boid in boids.Where(boid => boid != this.Boid))
        {
            Vector3 diff = boid.transform.position - this.transform.position;
            if (diff.magnitude < Radius)
            {
                average += diff;
                found += 1;
            }
        }

        if (found > 0)
        {
            average /= found;
            Boid.velocity += Vector3.Lerp(Vector3.zero, average, average.magnitude / Radius);
        }
        */
    }
}
