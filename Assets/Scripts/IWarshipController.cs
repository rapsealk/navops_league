using UnityEngine;

public interface IWarshipController
{
    Transform GetTransform();

    Warship GetOpponent();
}
