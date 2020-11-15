using UnityEngine;

public class RandomAgent
{
    public int GetAction()
    {
        return Random.Range(0, 6);
    }
}
