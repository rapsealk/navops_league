using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Shell : MonoBehaviour
{
    public ParticleSystem m_TrailParticleSystem;
    public ParticleSystem m_AfterburnerParticleSystem;

    // Start is called before the first frame update
    void Start()
    {
        m_TrailParticleSystem.Play();
        m_AfterburnerParticleSystem.Play();
    }

    // Update is called once per frame
    void Update()
    {
        if (transform.position.y < 0f)
        {
            Debug.Log($"{GetType().Name}.transform.position: {transform.position} ({transform.position.magnitude})");
            Destroy(gameObject);
        }
    }

    private void OnCollisionEnter(Collision collision)
    {
        Debug.Log($"{GetType().Name}.OnCollisionEnter: {collision.collider} {transform.position} ({transform.position.magnitude})");
        Destroy(gameObject);
    }
}
