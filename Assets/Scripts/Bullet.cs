using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Bullet : MonoBehaviour
{
    //public GameObject m_BattleShip;

    // Start is called before the first frame update
    void Start()
    {
        //Physics.IgnoreCollision(GetComponent<Collider>(), m_BattleShip.GetComponent<Collider>());
    }

    // Update is called once per frame
    void Update()
    {
        if (transform.position.y < 0)
        {
            Destroy(gameObject);
        }
    }

    /*
    void OnCollisionEnter(Collision collision)
    {
        if (collision.collider.tag == "Battleship")
        {
            Physics.IgnoreCollision(GetComponent<Collider>(), collision.collider);
        }
        Destroy(gameObject);
    }
    */

    void OnTriggerEnter(Collider collider)
    {
        Destroy(gameObject);
    }
}
