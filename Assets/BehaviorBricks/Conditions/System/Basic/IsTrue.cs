using Pada1.BBCore.Framework;
using Pada1.BBCore;
using UnityEngine;

namespace BBCore.Conditions
{
    /// <summary>
    /// It is a basic condition to check if Booleans have the same value.
    /// </summary>
    [Condition("Basic/IsTrue")]
    [Help("Check if variable is true")]
    public class IsTrue : ConditionBase
    {
        ///<value>Input First Boolean Parameter.</value>
        [InParam("Boolean")]
        [Help("Boolean to be checked")]
        public bool boolean;

        /// <summary>
        /// Checks whether two booleans have the same value.
        /// </summary>
        /// <returns>the value of compare first boolean with the second boolean.</returns>
		public override bool Check()
        {
            return boolean;
        }
    }
}