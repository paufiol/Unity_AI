using Pada1.BBCore.Framework;
using Pada1.BBCore;
using UnityEngine;

namespace BBCore.Conditions
{
    /// <summary>
    /// It is a basic condition to check if Booleans have the same value.
    /// </summary>
    [Condition("Basic/IsOutOfAmmo")]
    [Help("Checks if run out of ammo")]
    public class IsOutOfAmmo : ConditionBase
    {
        ///<value>Input First Boolean Parameter.</value>
        [InParam("Ammo")]
        [Help("First value to be compared")]
        public int ammo;

        /// <summary>
        /// Checks whether two booleans have the same value.
        /// </summary>
        /// <returns>the value of compare first boolean with the second boolean.</returns>
		public override bool Check()
        {
            //Debug.Log("Has " + ammo.ToString() + " ammo left");
            return ammo <= 0;
        }
    }
}