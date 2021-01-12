using Pada1.BBCore;
using Pada1.BBCore.Tasks;
using UnityEngine;

namespace BBUnity.Actions
{
    /// <summary>
    /// Makes the tank wander
    /// </summary>
    [Action("Navigation/Patrol")]
    [Help("make an object patrol")]
    public class Patrol : GOAction
    {

        [InParam("IsPatroller")]
        public bool IsPatroller;

        [InParam("CurrentDestination")]
        public int CurrentDestination;

        public Transform[] PatrolPoints;

        private UnityEngine.AI.NavMeshAgent Agent;

        [OutParam("CurrentDestinationReturn")]
        public int CurrentDestinationReturn;

        public override void OnStart()
        {
            if (IsPatroller) //Only do this is it's thw wandering tank
            {
                PatrolPoints = GetPatrolPoints("PatrolPoint");

                Agent = gameObject.GetComponent<UnityEngine.AI.NavMeshAgent>();

                if (!Agent.pathPending && Agent.remainingDistance < 0.5f)
                    NextPatrolPoint();

            }

            CurrentDestinationReturn = CurrentDestination;

        }

        void NextPatrolPoint()
        {
            if (PatrolPoints.Length == 0)
            {
                Debug.Log("There are no patrolling points");
                return;
            }

            Agent.destination = PatrolPoints[CurrentDestination].position;
            CurrentDestination = (CurrentDestination + 1) % PatrolPoints.Length;
        }

        Transform[] GetPatrolPoints(string tag)
        {
            var ret = new System.Collections.Generic.List<Transform>();
            var objects = Object.FindObjectsOfType(typeof(GameObject)) as GameObject[];
            for (int i = 0; i < objects.Length; i++)
            {
                if (objects[i].tag == tag)
                {
                    ret.Add(objects[i].transform);
                    //Debug.Log("Patrolling Point Found: " + objects[i].name);
                }
            }
            return ret.ToArray();
        }

        public override TaskStatus OnUpdate()
        {
            return TaskStatus.COMPLETED;
        }
    }
}
