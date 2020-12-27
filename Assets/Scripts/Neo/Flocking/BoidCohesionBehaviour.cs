using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

[RequireComponent(typeof(Boid))]
public class BoidCohesionBehaviour : MonoBehaviour
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
        Boid[] boids = FindObjectsOfType<Boid>();   // TODO: Filter
        Vector3 average = Vector3.zero;
        uint found = 0;

        foreach (var boid in boids.Where(boid => boid != this.Boid))
        {
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
            Boid.velocity += Vector3.Lerp(Boid.velocity, average, Time.deltaTime);
        }
    }
}
