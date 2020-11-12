using System.Collections;
using System.Collections.Generic;
using System.Threading;
using UnityEngine;

public class CameraController : MonoBehaviour
{
    public Transform m_Target;
    public float m_MaxTiltAngle = 30f;

    private float m_RotationAngle = 0f;
    private float m_TiltAngle = 0f;
    //private float moveSpeed = 0.5f;
    //private float scrollSpeed = 10f;

    // Start is called before the first frame update
    void Start()
    {
        Cursor.visible = false;
        Cursor.lockState = CursorLockMode.Locked;
    }

    // Update is called once per frame
    void Update()
    {
        float horizontal = Input.GetAxis("Mouse X") * Time.deltaTime;
        float vertical = -Input.GetAxis("Mouse Y") * Time.deltaTime;

        m_RotationAngle = (m_RotationAngle + horizontal * 100) % 360;
        float radians = m_RotationAngle / 180 * Mathf.PI;

        m_TiltAngle = (m_TiltAngle + vertical * 100);
        if (Mathf.Abs(m_TiltAngle) > m_MaxTiltAngle)
        {
            m_TiltAngle = Mathf.Sign(m_TiltAngle) * m_MaxTiltAngle;
        }

        float x = Mathf.Cos(radians);
        float y = Mathf.Sin(radians);
        transform.position = new Vector3(y, 3f, x);
        transform.rotation = Quaternion.Euler(new Vector3(m_TiltAngle, m_RotationAngle, 0f));

        // Debug.Log($"Rotation: {radians} ({x}, {y})");

        /*
        if (horizontal != 0 || vertical != 0)
        {
            transform.Rotate(vertical, horizontal, 0);
        }
        */

        /*
        if (Input.GetAxis("Mouse ScrollWheel") != 0)
        {
            transform.position += scrollSpeed * new Vector3(0, -Input.GetAxis("Mouse ScrollWheel"), 0) * Time.deltaTime;
        }*/

        //Debug.Log($"CameraController.Update({horizontal}, {vertical})");

        transform.position = m_Target.position + new Vector3(0, 5, -5);
        //transform.Rotate(m_Target.rotation.eulerAngles, relativeTo: Space.Self);
    }
}
