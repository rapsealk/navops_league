using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Torpedo : MonoBehaviour
{
    [Tooltip("Velocity in kilometers/h.")]
    public float Velocity;
    [Tooltip("Maximum range distance in kilometers.")]
    public float MaxRange;

    private Vector3 launchedPosition;

    // Start is called before the first frame update
    void Start()
    {
        launchedPosition = transform.position;
    }

    // Update is called once per frame
    void Update()
    {
        transform.position += transform.up * Velocity * Time.deltaTime;

        if ((transform.position - launchedPosition).magnitude >= MaxRange)
        {
            Destroy(gameObject);
        }
    }

    private void OnCollisionEnter(Collision collision)
    {
        Debug.Log($"Torpedo.OnCollisionEnter(collision: {collision})");

        Destroy(gameObject);
    }
}
