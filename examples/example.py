from ssp.solver import DGSSPSolver

def main():
    A=[3,-1,2,5]
    t=4
    solver=DGSSPSolver(A,t,assembly_type='FullQFT')

    qc=solver.solve(iterations=1)

    print(qc.draw())

if __name__=='__main__':
    main()