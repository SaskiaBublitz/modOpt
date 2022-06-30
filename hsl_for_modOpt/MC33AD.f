* COPYRIGHT (c) 1987 AEA Technology
* Original date 21 Dec 1992
C       Toolpack tool decs employed.
C       Args M, N and NZ of MC33C/CD and MC33D/DD not used - dummy
C         assignments inserted for compiler
C       Label 100 removed from MC33A/AD.
C       IW(1) tested for > 0, error if not.
C       MAXTAL set to zero initially.
C
C January 2002: use of MC20 replaced by MC59
C 13/3/02 Cosmetic changes applied to reduce single/double differences
C
C *********************  MC33A ****************************************
C
C 12th July 2004 Version 1.0.0. Version numbering added.

      SUBROUTINE MC33AD(M,N,NZI,A,IRN,JCN,NZO,ITYPE,IP,IQ,IPROF,IFLAG,
     +                  IW,IW1,IERR)
C     .. Scalar Arguments ..
      INTEGER IERR,ITYPE,M,N,NZI,NZO
C     ..
C     .. Array Arguments ..
      DOUBLE PRECISION A(NZI)
      INTEGER IFLAG(3),IP(*),IPROF(*),IQ(*),IRN(*),IW(*),IW1(*),JCN(*)
C     ..
C     .. Local Scalars ..
      DOUBLE PRECISION ATEMP
      INTEGER BLM,FLM,FLT,HEAD,I,I1,II,IPINV,IQINV,IROW,IROWST,J,JCOLST,
     +        JCSBND,JFLAG,JTYPE,K,KZ,KZL,LENCOL,LENROW,MINR,TALLY
C     ..
C     .. Local Arrays ..
      INTEGER ICNT59(10),INFO59(10)
C     ..
C     .. External Subroutines ..
      EXTERNAL MC59AD
C     ..,MC33BD
C     .. Intrinsic Functions ..
      INTRINSIC ABS
C     ..
C     .. Executable Statements ..
      IERR = 0
      NZO = 0
C
C  CHECK INPUT PARAMETERS. SET IERR AND RETURN IF ERROR
C
      IF (M.LE.0) THEN
        IERR = -1
        GO TO 1000

      END IF

      IF (N.LE.0) THEN
        IERR = -2
        GO TO 1000

      END IF

      IF (NZI.LE.0) THEN
        IERR = -3
        GO TO 1000

      END IF
C
C  JUMP IF MATRIX ALREADY SORTED
C
      NZO = NZI
      IF (ITYPE.LE.0) GO TO 1001
      NZO = 0
      DO 50 K = 1,NZI
        I = IRN(K)
        J = JCN(K)
        ATEMP = A(K)
        IF (J.LE.N) THEN
          NZO = NZO + 1
          IRN(NZO) = I
          JCN(NZO) = J
          A(NZO) = ATEMP
        END IF

   50 CONTINUE
      IERR = NZI - NZO
C      CALL MC20AD(N,NZO,A,IRN,IW,JCN,0)
C No data checking
      ICNT59(1) = 1
C Input arbitrary order, output by columns
      ICNT59(2) = 0
C Apply ordering to A
      ICNT59(3) = 0
C Switch off error and warning messages
      ICNT59(4) = -1
      ICNT59(5) = -1
C The matrix is unsymmetric
      ICNT59(6) = 0
      CALL MC59AD(ICNT59,N,M,NZO,IRN,NZO,JCN,NZO,A,N+1,IW,
     +            MAX(N,M)+1,IW1,INFO59)
 1001 IW(N+1) = NZO + 1
      IF (IW(1).LE.0) THEN
        IERR = -4
        RETURN

      END IF

      DO 40 I = 1,N
        IF (IW(I).GT.IW(I+1)) THEN
          IERR = -4
          RETURN

        END IF

   40 CONTINUE
      KZ = 0
      DO 30 I = 1,M
        IW1(N+I) = 0
   30 CONTINUE
      DO 10 J = 1,N
        LENCOL = IW(J+1) - IW(J)
C
C  REMOVE DUPLICATE
C
        I1 = IW(J)
        KZL = KZ
        DO 20 II = I1,I1 + LENCOL - 1
          I = IRN(II)
          ATEMP = A(II)
          IF (IW1(N+I).EQ.J) GO TO 20
          IF (I.LE.M) THEN
            KZ = KZ + 1
            IRN(KZ) = I
            A(KZ) = ATEMP
            IW1(N+I) = J

          ELSE
            IERR = IERR + 1
          END IF

   20   CONTINUE
        IW1(J) = KZ - KZL
   10 CONTINUE
      NZO = KZ
C
C  DIVIDE WORK SPACE
C
      LENCOL = 1
      JCOLST = 1
      IROWST = JCOLST + N
      LENROW = LENCOL + N
      IPINV = LENROW + M
      IQINV = IPINV + M
      MINR = IQINV + N
      TALLY = MINR + N
      HEAD = TALLY + N
      FLM = HEAD + N
      FLT = FLM + N
      BLM = FLT + N
      JFLAG = BLM + N
      IROW = JFLAG + N
      JTYPE = 5
      IF (ABS(ITYPE).EQ.3) JTYPE = 3
      CALL MC33BD(M,N,NZO,IRN,IW1(LENCOL),IPROF,JCSBND,IFLAG(2),
     +            IFLAG(1),IP,IQ,IW(JCOLST),JCN,IW(IROWST),IW1(LENROW),
     +            IW1(IPINV),IW1(IQINV),IW1(MINR),IW1(TALLY),IW1(HEAD),
     +            IW1(FLM),IW1(FLT),IW1(BLM),IW1(JFLAG),IW1(IROW),JTYPE)
      IFLAG(3) = IFLAG(2) - JCSBND
 1000 RETURN

      END
C
C***********************************************************************
C
      SUBROUTINE MC33BD(M,N,NZ,IRN,LENCOL,IPROF,JCSBND,JBEND,IRSBND,IP,
     +                  IQ,JCOLST,JCN,IROWST,LENROW,IPINV,IQINV,MINR,
     +                  TALLY,HEAD,FLM,FLT,BLM,JFLAG,IROW,JTYPE)
C
C                           INPUT
C
C      M IS AN INTEGER VARIABLE SET BY THE USER TO THE NUMBER OF ROWS
C                   IN THE MATRIX. IT IS UNCHANGED BY THE SUBROUTINE.
C      N IS AN INTEGER VARIABLE SET BY THE USER TO THE NUMBER OF COLUMNS
C                   IN THE MATRIX. IT IS UNCHANGED BY THE SUBROUTINE.
C     NZ IS AN INTEGER VARIABLE SET BY THE USER TO THE NUMBER OF ENTRIES
C                   IN THE MATRIX. IT IS UNCHANGED BY THE SUBROUTINE.
C    IRN IS AN INTEGER*2 ARRAY OF LENGTH NZ, SET BY THE USER TO THE
C        ROW INDICES OF THE ENTRIES, ORDERED SO THAT THE ENTRIES
C        BELONGING TO A SINGLE COLUMN ARE CONTIGUOUS, BUT THE ORDERING
C        OF ENTRIES WITHIN EACH COLUMN IS UNIMPORTANT. THE ENTRIES
C        OF COLUMN J PRECEDE THOSE OF COLUMN J+1 (J = 1, ..., N-1).
C        IT IS NOT ALTERED BY THRE ROUTINE. THE ENTRIES MUST BE >0 .
C LENCOL IS AN INTEGER*2 ARRAY OF LENGTH N THAT MUST BE SET BY THE USER
C        SO THAT LENCOL(J) IS THE NUMBER OF ENTRIES IN COLUMN J OF THE
C        ORIGINAL MATRIX. IT IS UNALTERED BY THE ROUTINE.
C JTYPE  IS AN INTEGER VARIABLE. IF THE VALUE OF JTYPE IS 3 THEN THE
C        P3 (PREASSIGNED PIVOT PROCEDURE, WHICH PRODUCES THE NESTED
C        BORDERED TRIANGULAR FORM) IS INVOKED. ANY OTHER VALUE WILL
C        EMPLOY P5 (PRECAUTIONARY PARTITIONED PREASSIGNED PIVOT
C        PROCEDURE, WHICH PRODUCES THE ORDINARY BORDERED BLOCK
C        TRIANGULAR FORM,WITH THE BLOCK COMPLETELY FULL). THIS
C        ARGUMENT IS NOT ALTERED BY THE ROUTINE.
C
C
C                     OUTPUT
C
C    IPROF  IS AN INTEGER*2 ARRAY OF LENGTH N WHICH NEED NOT BE SET BY
C           THE USER. IT GIVES THE PROFILE OF THE MATRIX
C  IRSBND   IS AN INTEGER VARIABLE WHICH NEED NOT BE SET BY THE USER.
C           IT WILL BE SET BY THE ROUTINE TO THE INDEX IN THE PERMUTED
C           MATRIX OF THE LAST ROW IN THE BORDER.
C           N.B. THE INDEX OF THE FIRST ROW IN THE BORDER IS
C           THE SAME AS THAT OF THE FIRST COLUMN IN THE BORDER.
C   IP,IQ   ARE INTEGER*2 ARRAYS OF LENGTH N AND M WHICH NEED NOT BE SET
C           BY THE USER. THEY ARE SET BY THE ROUTINE SO THAT THEY
C           ARE THE ROW AND COLUMN PERMUTATION MATRICES, IN PACKED FORM,
C           SO THAT ROW IP(I) OF THE ORIGINAL MATRIX IS ROW
C           I OF THE PERMUTED FORM, AND LIKEWISE WITH IQ FOR THE COLS.
C
C                   WORK ARRAYS
C
C JCOLST(J) IS THE POSTION IN ARRAY IRN OF THE FIRST ENTRY IN
C           COLUMN J.
C    JCN IS AN INTEGER*2 ARRAY OF LENGTH NZ, SET BY THE ROUTINE SO THAT
C        JCN(I) IS THE COLUMN INDEX OF THE ITH ENTRY OF THE ORIGINAL
C        MATRIX. THOSE BELONGING TO A SINGLE ROW ARE CONTIGUOUS,
C        AND THE ENTRIES WITHIN EACH ROW ARE IN ASCENDING ORDER
C        THE ENTRIES OF ROW I PRECEDE THOSE OF ROW I+1 (I = 1, ...
C        N-1)
C IROWST(I) IS THE POSITION IN ARRAY JCN OF THE FIRST ENTRY IN ROW I.
C LENROW(I) IS THE NUMBER OF ENTRIES IN ROW I OF THE ACTIVE SUB-MATRIX.
C IPINV,IQINV ARE THE INVERSE PERMUTATIONS TO IP & IQ.
C   MINR(J) IS THE LENGTH OF THE SHORTEST ROW WITH AN ENTRY IN COLUMN J.
C  TALLY(J) IS THE NUMBER OF ROWS OF LENGTH MINR(J) WITH ENTRIES IN
C                                                              COLUMN J.
C   HEAD(M) IS THE FIRST COLUMN  WITH THE GREATEST TALLY VALUE AMONG
C           THOSE WITH MINR VALUE OF M, OR ZERO IF NO MINR VALUE IS M.
C    FLM(J) IS THE NEXT COLUMN WITH THE SAME MINR AND TALLY VALUES AS
C           COLUMN J, OR ZERO IF NONE.
C    FLT(J) FOR A COLUMN J THAT IS THE FIRST COLUMN IN AN FLM LIST,IS
C           THE FIRST COLUMN WITH THE SAME MINR VALUE AND NEXT LOWER
C           TALLY VALUE, OR ZERO IF NONE. IT IS ALSO ZERO IF J IS NOT
C           THE FIRST IN AN FLM LIST.
C    BLM(J) IS THE PREVIOUS COLUMN WITH THE SAME MINR AND TALLY VALUES
C           AS COLUMN J, OR THE NEGATION OF THE INDEX OF THE PREVIOUS
C           COLUMN WITH THE SAME MINR VALUE AND GREATER TALLY VALUE, OR
C           ZERO IF NEITHER.
C  JFLAG()  IS A FLAG USED WHEN UPDATING THE LINKED LISTS TO ENSURE THAT
C           NO COLUMN IS HANDLED TWICE, AND TO INDICATE WHETHER OR NOT
C           A COLUMN IS IN THE ACTIVE SUBMATRIX.
C  IROW()   IS A LIST OF ROWS IN THE COLUMN, JCOL, TO BE MOVED.
C
C  JCSBST IS THE INDEX OF THE FIRST COLUMN IN THE ACTIVE SUBMATRIX.
C   JBEND IS THE INDEX OF THE LAST COLUMN IN THE BORDER.
C  JCSBND IS THE INDEX OF THE LAST COLUMN IN THE ACTIVE SUBMATRIX.
C  IRSBST IS THE INDEX OF THE FIRST ROW TIN THE ACTIVE SUBMATRIX.
C  MINROW IS THE MINIMUM ROW LENGTH.
C  MCONST IS THE VALUE OF MINROW SET AFTER THE LAST CALL TO MC33C/D.
C  MNROW2 IS THE SECOND LOWEST ROW LENGTH FOR ROWS IN COLUMNS OF JFLAG.
C  JTS    IS THE GREATEST VALUE OF TALLY FOR COLUMNS WITH SMALLEST MINR.
C********* INITIALIZE VARIABLES***********
C     .. Scalar Arguments ..
      INTEGER IRSBND,JBEND,JCSBND,JTYPE,M,N,NZ
C     ..
C     .. Array Arguments ..
      INTEGER BLM(*),FLM(*),FLT(*),HEAD(*),IP(*),IPINV(*),IPROF(*),
     +        IQ(*),IQINV(*),IRN(*),IROW(*),IROWST(*),JCN(*),JCOLST(*),
     +        JFLAG(*),LENCOL(*),LENROW(*),MINR(*),TALLY(*)
C     ..
C     .. Local Scalars ..
      INTEGER I,IBACK,IFWD,II,IOLDMR,IPJ,IR,IRSBST,IT,ITEMP,J,JCOL,
     +        JCSBST,JDUMMY,JFWD,JJ,JTALY,JTEMP,JTI,JTS,JX,K1,KK,KTEMP,
     +        L,LOOP,LTEMP,MAXLEN,MAXTAL,MCONST,MIN1,MIN2,MINROW,MIROW,
     +        MR,NBLK
C     ..
C     .. External Subroutines ..
      EXTERNAL MC33CD,MC33DD
C     ..
C     .. Intrinsic Functions ..
      INTRINSIC ABS
C     ..
C     .. Executable Statements ..
      DO 10 J = 1,N
        IPROF(J) = 0
        FLM(J) = 0
        BLM(J) = 0
        JFLAG(J) = 0
        FLT(J) = 0
        HEAD(J) = 0
        TALLY(J) = 0
        MINR(J) = 0
        IQ(J) = J
        IQINV(J) = J
   10 CONTINUE
      DO 11 I = 1,M
        LENROW(I) = 0
        IROW(I) = 0
        IP(I) = I
        IPINV(I) = I
   11 CONTINUE
      NBLK = 0
      IRSBST = 1
      IRSBND = M
      JCSBST = 1
      JCSBND = N
      MCONST = 0
      MAXTAL = 0
C****** COUNT THE THE NUMBER OF ENTRIES IN EACH ROW ************
      DO 20 I = 1,NZ
        IR = IRN(I)
        LENROW(IR) = LENROW(IR) + 1
   20 CONTINUE
C******* ACCUMULATE THE ROW AND COLUMN COUNTS AND MOVE THOSE WITH NO ***
C******* ENTRIES toend OF THE MATRIX *******
      IROWST(1) = 1
      JCOLST(1) = 1
      IF (M.GT.1) THEN
        DO 30 I = 1,M - 1
          IROWST(I+1) = IROWST(I) + LENROW(I)
          IF (LENROW(I).EQ.0) THEN
C*** MOVE ROW WITH NO ENTRIES OF MATRIX *****
            ITEMP = IRSBND
            JTEMP = IPINV(I)
            KTEMP = IP(ITEMP)
            LTEMP = IP(JTEMP)
            IP(ITEMP) = LTEMP
            IP(JTEMP) = KTEMP
            IPINV(KTEMP) = JTEMP
            IPINV(LTEMP) = ITEMP
            IRSBND = IRSBND - 1
          END IF

          LENROW(I) = 0
   30   CONTINUE
      END IF

      IF (LENROW(M).EQ.0) IRSBND = IRSBND - 1
      LENROW(M) = 0
      IF (N.GT.1) THEN
        DO 35 J = 1,N - 1
          JCOLST(J+1) = JCOLST(J) + LENCOL(J)
          IF (LENCOL(J).EQ.0) THEN
C*** MOVE COL WITH NO ENTRIES TO END OF MATRIX *****
            ITEMP = JCSBND
            JTEMP = IQINV(J)
            KTEMP = IQ(ITEMP)
            LTEMP = IQ(JTEMP)
            IQ(ITEMP) = LTEMP
            IQ(JTEMP) = KTEMP
            IQINV(KTEMP) = JTEMP
            IQINV(LTEMP) = ITEMP
            JCSBND = JCSBND - 1
            JFLAG(J) = N + 1
          END IF

   35   CONTINUE
      END IF

      IF (LENCOL(N).EQ.0) JCSBND = JCSBND - 1
      JBEND = JCSBND
C
C  IF N OR M EQUALS 1 THEN RETURN
C
      IF (N.EQ.1) THEN
        IPROF(1) = 1
        RETURN

      END IF

      IF (M.EQ.1) THEN
        DO 31 I = 1,JBEND
          IPROF(I) = 1
   31   CONTINUE
        JCSBND = 1
        RETURN

      END IF
C
C******* GENERATE JCN FROM IRN,IROWST *******
      DO 50 J = 1,N
        DO 40 I = JCOLST(J),JCOLST(J) + LENCOL(J) - 1
          L = IRN(I)
          JCN(IROWST(L)+LENROW(L)) = J
          LENROW(L) = LENROW(L) + 1
   40   CONTINUE
   50 CONTINUE
C
C SET UP THE LINKED-LISTS
      DO 160 J = 1,N
        IF (LENCOL(J).EQ.0) GO TO 160
C**********FIND MINR & TALLY FOR COLUMN J ********
        MIN1 = N + 1
        JTALY = 0
        DO 100 I = JCOLST(J),JCOLST(J) + LENCOL(J) - 1
          IF (LENROW(IRN(I)).LE.MIN1) THEN
            IF (LENROW(IRN(I)).LT.MIN1) THEN
              JTALY = 1
              MIN1 = LENROW(IRN(I))

            ELSE
              JTALY = JTALY + 1
            END IF

          END IF

  100   CONTINUE
        TALLY(J) = JTALY
        MINR(J) = MIN1
C
C******* INSERT J IN LINKED-LIST FOR MINR/TALLY. THE LIST IS ORDERED
C******* IN DECREASING VALUE OF THE TALLY, AS DESCRIBED ABOVE.
C**FIND WHERE TO INSERT J*******
        IF (HEAD(MIN1).EQ.0) GO TO 120
        JX = HEAD(MIN1)
        DO 110 JDUMMY = 1,JBEND
          IF (TALLY(JX).EQ.JTALY) GO TO 130
          IF (TALLY(JX).LT.JTALY) GO TO 140
          IF (FLT(JX).EQ.0) GO TO 150
          JX = FLT(JX)
  110   CONTINUE
C******** PLACE COLUMN J IN HEADER FOR M=MIN1 *******
  120   HEAD(MIN1) = J
        GO TO 160
C**** INSERT J AHEAD OF COLUMN JX IN FLM/BLM ********
  130   IBACK = BLM(JX)
        FLM(J) = JX
        BLM(JX) = J
        BLM(J) = IBACK
        FLT(J) = FLT(JX)
        FLT(JX) = 0
        IF (IBACK.LT.0) FLT(-IBACK) = J
        IF (IBACK.EQ.0) HEAD(MIN1) = J
        IF (FLT(J).GT.0) BLM(FLT(J)) = -J
        GO TO 160
C**** INSERT J AHEAD OF JX IN FLT/BLM
  140   IBACK = BLM(JX)
        FLT(J) = JX
        BLM(JX) = -J
        BLM(J) = IBACK
        IF (IBACK.LT.0) FLT(-IBACK) = J
        IF (IBACK.EQ.0) HEAD(MIN1) = J
        GO TO 160
C******* INSERT J AT END OF FLT/BLM *********
  150   FLT(JX) = J
        BLM(J) = -JX
  160 CONTINUE
      MINROW = 0
C
C
C******** MAIN LOOP *********
C FINDS WHICH COLUMN TO MOVE TO THE BACK OF THE ACTIVE SUBMATRIX,
C EXCLUDES IT AND ALTERS THE LINKED-LISTS ACCORDINGLY, AND CALLS THE
C APPROPRIATE ASSIGN PIVOTS ROUTINE.
      DO 1000 LOOP = 1,JBEND
C******* FIND FIRST MINR AND ITS COLUMN INDEX ************
        IF (MINROW.GT.0) GO TO 200
        DO 180 MINROW = 1,N
          IF (HEAD(MINROW).NE.0) GO TO 190
  180   CONTINUE
  190   MCONST = MINROW
  200   JCOL = HEAD(MINROW)
        JTS = TALLY(JCOL)
        JTI = JCOL
C*** IF NO OTHER COLUMNS HAVE THE SAME MINR AND TALLY VALUES ***
C***               JCOL IS MOVED                             ***
        IF (FLM(JTI).EQ.0) GO TO 500
C
C TIE-BREAKING
        MAXLEN = -1
C*******  MAXLEN IS THE HIGHEST LENCOL OF COLUMNS WITH SAME MINR AND
C*******  TALLY VALUES ********
        IF (JTS.NE.1) GO TO 400
C**** IF TALLY = 1 THE FOLLOWING TIE-BREAK MUST BE DONE : ******
C****** FOR EACH COLUMN WITH MINR=MINROW AND TALLY=1 (SCAN MINR LIST)
C****** FIND 2ND MIN ROW LENGTH, AND FIND THE COLUMN WITH THE
C****** MAXIMUM NUMBER OF SUCH ROWS. THIS COLUMN BECOMES JCOL*******
        MIN2 = N + 1
        DO 320 JDUMMY = 1,JBEND
          JTALY = 0
          DO 310 J = JCOLST(JTI),JCOLST(JTI) + LENCOL(JTI) - 1
            L = IRN(J)
            IF (LENROW(L).LE.MIN2 .AND. LENROW(L).GT.MINROW) THEN
              IF (LENROW(L).LT.MIN2) THEN
                MIN2 = LENROW(L)
                JTALY = 1
                MAXTAL = -1

              ELSE
                JTALY = JTALY + 1
              END IF

            END IF

  310     CONTINUE
          IF (JTALY.GT.MAXTAL) THEN
            MAXTAL = JTALY
            MAXLEN = LENCOL(JTI)
            JCOL = JTI

          ELSE IF (JTALY.EQ.MAXTAL .AND. LENCOL(JTI).GT.MAXLEN) THEN
            MAXLEN = LENCOL(JTI)
            JCOL = JTI
          END IF

          JTI = FLM(JTI)
          IF (JTI.EQ.0) GO TO 500
  320   CONTINUE
C********* FIND LARGEST LENCOL OF TIEING COLUMNS ********
  400   DO 410 JDUMMY = 1,JBEND
          IF (LENCOL(JTI).GT.MAXLEN) THEN
            MAXLEN = LENCOL(JTI)
            JCOL = JTI
          END IF

          JTI = FLM(JTI)
          IF (JTI.EQ.0) GO TO 500
  410   CONTINUE
C
C******* COLUMN JCOL HAS NOW BEEN CHOSEN AND IS READY TO BE PERMUTED***
  500   JFLAG(JCOL) = N + 1
C******** EXPAND JCOL INTO IROW******
        DO 510 J = JCOLST(JCOL),JCOLST(JCOL) + LENCOL(JCOL) - 1
          IROW(IRN(J)) = LOOP
          IF (LENROW(IRN(J)).EQ.MINROW) IROW(IRN(J)) = -LOOP
  510   CONTINUE
C
C FIND THE ROWS WHOSE MINR OR TALLY WILL BE ALTERED BY EXCLUDING COLUMN
C JCOL AND ALTER THE LINKED-LISTS FOR EACH
C
        DO 700 II = JCOLST(JCOL),JCOLST(JCOL) + LENCOL(JCOL) - 1
C**** FOR EACH ROW IN JCOL FIND THE COLUMNS ******
          JJ = IRN(II)
          K1 = NZ
          IF (JJ.NE.M) K1 = IROWST(JJ+1) - 1
          DO 695 J = IROWST(JJ),K1
            IPJ = JCN(J)
            IF (JFLAG(IPJ).GE.LOOP) GO TO 695
            L = 0
            KK = 0
            DO 520 I = JCOLST(IPJ),JCOLST(IPJ) + LENCOL(IPJ) - 1
              MIROW = IROW(IRN(I))
              MIROW = ABS(MIROW)
              IF (MIROW.NE.LOOP) GO TO 520
C
C COUNT, IN L & KK, ROWS WITH ENTRIES IN BOTH COLUMN IPJ AND COLUMN JCOL
C THAT HAVE ROW COUNT MINR(IPJ) AND MINR(IPJ)+1
              IF (LENROW(IRN(I)).LE.MINR(IPJ)+1) THEN
                IF (LENROW(IRN(I)).EQ.MINR(IPJ)) THEN
                  L = L + 1

                ELSE
                  KK = KK + 1
                END IF

              END IF

  520       CONTINUE
C********* ALTER THE LINKED-LISTS ************
C***** ALTER TALLY AND MINR*******
            IOLDMR = MINR(IPJ)
            IF (L.NE.0) THEN
              TALLY(IPJ) = L
              MINR(IPJ) = IOLDMR - 1

            ELSE
              IF (KK.EQ.0) GO TO 690
              TALLY(IPJ) = TALLY(IPJ) + KK
            END IF
C********* DELETE IPJ FROM LIST OF OLD MINR/TALLY ********
            IBACK = BLM(IPJ)
            IFWD = FLM(IPJ)
            JFWD = FLT(IPJ)
            IF (IFWD.GT.0) THEN
              FLT(IFWD) = JFWD
              IF (JFWD.GT.0) BLM(JFWD) = -IFWD
            END IF

            IF (IBACK.EQ.0) THEN
              IF (IFWD.EQ.0) IFWD = FLT(IPJ)
              HEAD(IOLDMR) = IFWD

            ELSE IF (IBACK.GT.0) THEN
              FLM(IBACK) = IFWD

            ELSE
              IF (IFWD.EQ.0) IFWD = FLT(IPJ)
              FLT(-IBACK) = IFWD
            END IF

            IF (IFWD.GT.0) BLM(IFWD) = IBACK
C********* INSERT IPJ INTO LIST MIN=LL/TALLY ********
            MR = MINR(IPJ)
            IT = TALLY(IPJ)
C
C***FIND WHERE TO INSERT IPJ IN THE LISTS ***
C******SCAN ALONG LINK LIST TO FIND WHERE IPJ SHOULD GO AND INSERT****
            IF (HEAD(MR).EQ.0) GO TO 540
            JX = HEAD(MR)
  530       CONTINUE
            IF (TALLY(JX).EQ.IT) GO TO 550
            IF (TALLY(JX).LT.IT) GO TO 560
            IF (FLT(JX).EQ.0) GO TO 570
            JX = FLT(JX)
            GO TO 530
C***** IPJ INSERTED IN HEADER FOR  MINR=MR *****:
  540       HEAD(MR) = IPJ
            BLM(IPJ) = 0
            FLM(IPJ) = 0
            FLT(IPJ) = 0
            GO TO 690
C *** INSERT IPJ AHEAD OF COLUMN JX IN FLM/BLM ********
  550       IBACK = BLM(JX)
            FLM(IPJ) = JX
            BLM(JX) = IPJ
            BLM(IPJ) = IBACK
            FLT(IPJ) = FLT(JX)
            FLT(JX) = 0
            IF (IBACK.LT.0) FLT(-IBACK) = IPJ
            IF (IBACK.EQ.0) HEAD(MR) = IPJ
            IF (FLT(IPJ).GT.0) BLM(FLT(IPJ)) = -IPJ
            GO TO 690
C**** INSERT IPJ AHEAD OF JX IN FLT/BLM
  560       IBACK = BLM(JX)
            IFWD = FLM(IPJ)
            FLT(IPJ) = JX
            BLM(JX) = -IPJ
            BLM(IPJ) = IBACK
            FLM(IPJ) = 0
            IF (IBACK.LT.0) FLT(-IBACK) = IPJ
            IF (IBACK.EQ.0) HEAD(MR) = IPJ
            GO TO 690
C******INSERT IPJ ATend OF FLT/BLM ********
  570       FLT(JX) = IPJ
            BLM(IPJ) = -JX
            FLM(IPJ) = 0
            FLT(IPJ) = 0
C**** LABEL THE COLUMN AS HAVING BEEN DEALT WITH, SO THAT
C**** IT ISN'T ALTERED AGAIN *****
  690       JFLAG(IPJ) = LOOP
  695     CONTINUE
  700   CONTINUE
        DO 710 I = JCOLST(JCOL),JCOLST(JCOL) + LENCOL(JCOL) - 1
          LENROW(IRN(I)) = LENROW(IRN(I)) - 1
  710   CONTINUE
C**********DELETE JCOL FROM MINR LINKED-LIST**********
        IBACK = BLM(JCOL)
        IFWD = FLM(JCOL)
        JFWD = FLT(JCOL)
        IF (IFWD.GT.0) THEN
          FLT(IFWD) = JFWD
          IF (JFWD.GT.0) BLM(JFWD) = -IFWD
        END IF

        IF (IBACK.EQ.0) THEN
          IF (IFWD.EQ.0) IFWD = FLT(JCOL)
          HEAD(MINROW) = IFWD

        ELSE IF (IBACK.GT.0) THEN
          FLM(IBACK) = IFWD

        ELSE
          IF (IFWD.EQ.0) IFWD = FLT(JCOL)
          FLT(-IBACK) = IFWD
        END IF

        IF (IFWD.GT.0) BLM(IFWD) = IBACK
        FLM(JCOL) = 0
        BLM(JCOL) = 0
        FLT(JCOL) = 0
C***** REGISTER THE COLUMN SWAP ON THE COLUMN PERMUTATION MATRIX IQ ****
        IPROF(JCSBND) = NBLK + 1
        ITEMP = JCSBND
        JTEMP = IQINV(JCOL)
        KTEMP = IQ(ITEMP)
        LTEMP = IQ(JTEMP)
        IQ(ITEMP) = LTEMP
        IQ(JTEMP) = KTEMP
        IQINV(KTEMP) = JTEMP
        IQINV(LTEMP) = ITEMP
        JCSBND = JCSBND - 1
        IF (MINROW.EQ.1 .AND. JTYPE.EQ.3) CALL MC33DD(M,N,NZ,IROW,JTS,
     +      IPINV,JCSBND,JCSBST,JBEND,JCOLST,LENCOL,IRN,IRSBST,LOOP,IP,
     +      IQ,IQINV,IPROF,NBLK)
        IF (MINROW.EQ.1 .AND. JTYPE.NE.3) CALL MC33CD(M,N,NZ,IROW,JTS,
     +      MCONST,IPINV,JCSBND,IRN,JCOLST,LENCOL,JCSBST,IRSBST,LOOP,IP,
     +      IQ,IQINV,IPROF,NBLK)
        MINROW = MINROW - 1
 1000 CONTINUE
      RETURN

      END
C
C************************************************************:
C
      SUBROUTINE MC33CD(M,N,NZ,IROW,JTS,MCONST,IPINV,JCSBND,IRN,JCOLST,
     +                  LENCOL,JCSBST,IRSBST,LOOP,IP,IQ,IQINV,IPROF,
     +                  NBLK)
C
C***************** ASSIGN-PIVOTSSUBROUTINE FOR ******************
C*** PRECAUTIONARY PARTITIONED PREASSIGNED PIVOT PROCEDURE (P5) ***
C**** (ERISMAN, GRIMES, LEWIS AND POOLE)
C
C
C************* THE ARGUMENTS ARE THE SAME AS FOR MC33B *******
C
C     .. Scalar Arguments ..
      INTEGER IRSBST,JCSBND,JCSBST,JTS,LOOP,M,MCONST,N,NBLK,NZ
C     ..
C     .. Array Arguments ..
      INTEGER IP(*),IPINV(*),IPROF(*),IQ(*),IQINV(*),IRN(*),IROW(*),
     +        JCOLST(*),LENCOL(*)
C     ..
C     .. Local Scalars ..
      INTEGER II,IPTEMP,ITEMP,J,JJ,JTEMP,K,KTEMP,L,LTEMP,MINMS
C     ..
C     .. Intrinsic Functions ..
      INTRINSIC MIN
C     ..
C     .. Executable Statements ..
C      Args M, N and NZ not used - dummy assignments for compiler
      ITEMP = M
      ITEMP = N
      ITEMP = NZ
C
      MINMS = MIN(MCONST,JTS)
      II = IQ(JCSBND+1)
C
C  THE FIRST MINMS COLUMNS IN THE BORDER ARE MOVED AHEAD OF THE ACTIVE
C  SUBMATRIX
C
      DO 10 J = 1,MINMS
        ITEMP = JCSBND + J
        JTEMP = JCSBST + J - 1
        KTEMP = IQ(ITEMP)
        LTEMP = IQ(JTEMP)
        IQ(ITEMP) = LTEMP
        IQ(JTEMP) = KTEMP
        IQINV(KTEMP) = JTEMP
        IQINV(LTEMP) = ITEMP
C
C  THE ENTRIES OF IPROF ARE SET WITH VALUE NBLK+1. THIS IDENTIFIES THE
C  DIAGONAL BLOCK OF THE FINAL MATRIX
C
        IPTEMP = IPROF(JCSBND+J)
        IPROF(JCSBND+J) = IPROF(JCSBST+J-1)
        IPROF(JCSBST+J-1) = IPTEMP
   10 CONTINUE
      JCSBND = JCSBND + MINMS
      JCSBST = JCSBST + MINMS
C
C  PERMUTE THE ROWS SO THAT THE ROWS THAT HAVE JUST HAD COUNT 1 ARE THE
C  THE LEADING ROWS OF THE ACTIVE SUBMATRIX
C
      L = 1
      DO 20 JJ = JCOLST(II),JCOLST(II) + LENCOL(II) - 1
        IF (IROW(IRN(JJ)).NE.-LOOP) GO TO 20
        K = IRN(JJ)
        ITEMP = IRSBST
        JTEMP = IPINV(K)
        KTEMP = IP(ITEMP)
        LTEMP = IP(JTEMP)
        IP(ITEMP) = LTEMP
        IP(JTEMP) = KTEMP
        IPINV(KTEMP) = JTEMP
        IPINV(LTEMP) = ITEMP
        IRSBST = IRSBST + 1
        L = L + 1
        IF (L.GT.MINMS) GO TO 40
   20 CONTINUE
   40 CONTINUE
      NBLK = NBLK + MINMS
      RETURN

      END
C
C **********************************************************************
C
      SUBROUTINE MC33DD(M,N,NZ,IROW,JTS,IPINV,JCSBND,JCSBST,JBEND,
     +                  JCOLST,LENCOL,IRN,IRSBST,LOOP,IP,IQ,IQINV,IPROF,
     +                  NBLK)
C
C** ASSIGN-PIVOTS SUBROUTINE FOR PREASSIGNED PIVOTS PROCEDURE (P3) **
C*** (HELLERMAN AND RARICK)
C
C
C*********** THE ARGUMENTS ARE THE SAME AS IN MC33B ***************
C
C     .. Scalar Arguments ..
      INTEGER IRSBST,JBEND,JCSBND,JCSBST,JTS,LOOP,M,N,NBLK,NZ
C     ..
C     .. Array Arguments ..
      INTEGER IP(*),IPINV(*),IPROF(*),IQ(*),IQINV(*),IRN(*),IROW(*),
     +        JCOLST(*),LENCOL(*)
C     ..
C     .. Local Scalars ..
      INTEGER II,IPTEMP,ITEMP,J,J1,J2,JJ,JT,JTEMP,K,KTEMP,LTEMP
C     ..
C     .. Executable Statements ..
C      Args M, N and NZ not used - dummy assignments for compiler
      ITEMP = M
      ITEMP = N
      ITEMP = NZ
C
      DO 1000 J = 1,JTS
C***** MOVE THE FIRST COLUMN BEYOND THE ACTIVE SUBMATRIX TO THE FRONT
C***** OF THE ACTIVE SUBMATRIX ****
        IF (JCSBND+J.GT.JBEND) GO TO 3000
        ITEMP = JCSBND + J
        JTEMP = JCSBST + J - 1
        KTEMP = IQ(ITEMP)
        LTEMP = IQ(JTEMP)
        IQ(ITEMP) = LTEMP
        IQ(JTEMP) = KTEMP
        IQINV(KTEMP) = JTEMP
        IQINV(LTEMP) = ITEMP
C
C  THE ENTRIES OF IPROF ARE PERMUTED. THIS IDENTIFIES THE
C  DIAGONAL BLOCK OF THE FINAL MATRIX
C
        IPTEMP = IPROF(ITEMP)
        IPROF(ITEMP) = IPROF(JTEMP)
        IPROF(JTEMP) = IPTEMP
        NBLK = NBLK + 1
        II = IQ(JCSBST+J-1)
C****** PERMUTE A SHORT ROW (IROW=-LOOP, LENROW=MINROW)TO THE FRONT OF
C****** THE ACTIVE SUBMATRIX *******
        DO 20 JJ = JCOLST(II),JCOLST(II) + LENCOL(II) - 1
          IF (IROW(IRN(JJ)).NE.-LOOP) GO TO 20
          IROW(IRN(JJ)) = LOOP
          K = IRN(JJ)
          ITEMP = IRSBST + J - 1
          JTEMP = IPINV(K)
          KTEMP = IP(ITEMP)
          LTEMP = IP(JTEMP)
          IP(ITEMP) = LTEMP
          IP(JTEMP) = KTEMP
          IPINV(KTEMP) = JTEMP
          IPINV(LTEMP) = ITEMP
          GO TO 1000

   20   CONTINUE
        GO TO 2000

 1000 CONTINUE
      GO TO 3000
C********* MOVE THE COLUMN AHEAD OF THE ACTIVE SUBMATRIX BACK TO THE
C********* BORDER****
 2000 ITEMP = JCSBST + J - 1
      JTEMP = JCSBND + J
      KTEMP = IQ(ITEMP)
      LTEMP = IQ(JTEMP)
      IQ(ITEMP) = LTEMP
      IQ(JTEMP) = KTEMP
      IQINV(KTEMP) = JTEMP
      IQINV(LTEMP) = ITEMP
C
C  THE ENTRIES OF IPROF ARE PERMUTED. THIS IDENTIFIES THE RIGHT SIDE
C  DIAGONAL BLOCK OF THE BORDER IN THE FINAL MATRIX
C
      IPTEMP = IPROF(ITEMP)
      IPROF(ITEMP) = IPROF(JTEMP)
      IPROF(JTEMP) = IPTEMP
      NBLK = NBLK - 1
 3000 CONTINUE
C
C  CHECK IF THE FIRST COLUMN OF BORDER CAN BE MOVED IN FRONT
C  AT THE END OF THE PROCESS
C
      IF (LOOP.EQ.JBEND) THEN
        J = 2
        J1 = JCSBND + J - 1
        J2 = J1 + 1
        IF (J2.LE.JBEND .AND. IPROF(J2).EQ.IPROF(J1)) THEN
          JT = IQ(J2)
          DO 200 JJ = JCOLST(JT),JCOLST(JT) + LENCOL(JT) - 1
            K = IPINV(IRN(JJ))
            IF (K.EQ.J2) J = 3
  200     CONTINUE
        END IF

      END IF

      JCSBND = JCSBND + J - 1
      JCSBST = JCSBST + J - 1
C****** REMOVE THE LEADING J-1 ROWS FROM THE ACTIVE SUBMATRIX******
      IRSBST = IRSBST + (J-1)
      RETURN

      END

* COPYRIGHT (c) 1993 Council for the Central Laboratory
*                    of the Research Councils

C Original date 29 Jan 2001
C 29 January 2001. Modified from MC49 to be threadsafe.

C 12th July 2004 Version 1.0.0. Version numbering added.
C 28 February 2008. Version 1.0.1. Comments flowed to column 72.
C 21 September 2009. Version 1.0.2. Minor change to documentation.

      SUBROUTINE MC59AD(ICNTL,NC,NR,NE,IRN,LJCN,JCN,LA,A,LIP,IP,
     &                  LIW,IW,INFO)
C
C To sort the sparsity pattern of a matrix to an ordering by columns.
C There is an option for ordering the entries within each column by
C increasing row indices and an option for checking the user-supplied
C matrix entries for indices which are out-of-range or duplicated.
C
C ICNTL:  INTEGER array of length 10. Intent(IN). Used to specify
C         control parameters for thesubroutine.
C ICNTL(1): indicates whether the user-supplied matrix entries are to
C           be checked for duplicates, and out-of-range indices.
C           Note  simple checks are always performed.
C           ICNTL(1) = 0, data checking performed.
C           Otherwise, no data checking.
C ICNTL(2): indicates the ordering requested.
C           ICNTL(2) = 0, input is by rows and columns in arbitrary
C           order and the output is sorted by columns.
C           ICNTL(2) = 1, the output is also row ordered
C           within each column.
C           ICNTL(2) = 2, the input is already ordered by
C           columns and is to be row ordered within each column.
C           Values outside the range 0 to 2 are flagged as an error.
C ICNTL(3): indicates whether matrix entries are also being ordered.
C           ICNTL(3) = 0, matrix entries are ordered.
C           Otherwise, only the sparsity pattern is ordered
C           and the array A is not accessed by the routine.
C ICNTL(4): the unit number of the device to
C           which error messages are sent. Error messages
C           can be suppressed by setting ICNTL(4) < 0.
C ICNTL(5): the unit number of the device to
C           which warning messages are sent. Warning
C           messages can be suppressed by setting ICNTL(5) < 0.
C ICNTL(6)  indicates whether matrix symmetric. If unsymmetric, ICNTL(6)
C           must be set to 0.
C           If ICNTL(6) = -1 or 1, symmetric and only the lower
C           triangular part of the reordered matrix is returned.
C           If ICNTL(6) = -2 or 2, Hermitian and only the lower
C           triangular part of the reordered matrix is returned.
C           If error checks are performed (ICNTL(1) = 0)
C           and ICNTL(6)> 1 or 2, the values of duplicate
C           entries are added together; if ICNTL(6) < -1 or -2, the
C           value of the first occurrence of the entry is used.
C ICNTL(7) to ICNTL(10) are not currently accessed by the routine.
C
C NC:      INTEGER variable. Intent(IN). Must be set by the user
C          to the number of columns in the matrix.
C NR:      INTEGER variable. Intent(IN). Must be set by the user
C          to the number of rows in the matrix.
C NE:      INTEGER variable. Intent(IN). Must be set by the user
C          to the number of entries in the matrix.
C IRN: INTEGER array of length NE. Intent (INOUT). Must be set by the
C            user to hold the row indices of the entries in the matrix.
C          If ICNTL(2).NE.2, the entries may be in any order.
C          If ICNTL(2).EQ.2, the entries in column J must be in
C            positions IP(J) to IP(J+1)-1 of IRN. On exit, the row
C            indices are reordered so that the entries of a single
C            column are contiguous with column J preceding column J+1, J
C            = 1, 2, ..., NC-1, with no space between columns.
C          If ICNTL(2).EQ.0, the order within each column is arbitrary;
C            if ICNTL(2) = 1 or 2, the order within each column is by
C            increasing row indices.
C LJCN:    INTEGER variable. Intent(IN). Defines length array
C JCN:     INTEGER array of length LJCN. Intent (INOUT).
C          If ICNTL(2) = 0 or 1, JCN(K) must be set by the user
C          to the column index of the entry
C          whose row index is held in IRN(K), K = 1, 2, ..., NE.
C          On exit, the contents of this array  will have been altered.
C          If ICNTL(2) = 2, the array is not accessed.
C LA:      INTEGER variable. Intent(IN). Defines length of array
C          A.
C A:       is a REAL (DOUBLE PRECISION in the D version, INTEGER in
C          the I version, COMPLEX in the C version,
C          or COMPLEX"*"16 in the Z version) array of length LA.
C          Intent(INOUT).
C          If ICNTL(3).EQ.0, A(K) must be set by the user to
C          hold the value of the entry with row index IRN(K),
C          K = 1, 2, ..., NE. On exit, the array will have been
C          permuted in the same way as the array IRN.
C          If ICNTL(3).NE.0, the array is not accessed.
C LIP:     INTEGER variable. Intent(IN). Defines length of array
C          IP.
C IP:      INTEGER array of length LIP. Intent(INOUT). IP
C          need only be set by the user if ICNTL(2) = 2.
C          In this case, IP(J) holds the position in
C          the array IRN of the first entry in column J, J = 1, 2,
C          ..., NC, and IP(NC+1) is one greater than the number of
C          entries in the matrix.
C          In all cases, the array IP will have this meaning on exit
C          from thesubroutine and is altered when ICNTL(2) = 2 only
C          when ICNTL(1) =  0 and there are out-of-range
C          indices or duplicates.
C LIW:     INTEGER variable. Intent(IN). Defines length of array
C          IW.
C IW:      INTEGER array of length LIW. Intent(OUT). Used by the
C          routine as workspace.
C INFO:    INTEGER array of length 10.  Intent(OUT). On exit,
C          a negative value of INFO(1) is used to signal a fatal
C          error in the input data, a positive value of INFO(1)
C          indicates that a warning has been issued, and a
C          zero value is used to indicate a successful call.
C          In cases of error, further information is held in INFO(2).
C          For warnings, further information is
C          provided in INFO(3) to INFO(6).  INFO(7) to INFO(10) are not
C          currently used and are set to zero.
C          Possible nonzero values of INFO(1):
C         -1 -  The restriction ICNTL(2) = 0, 1, or 2 violated.
C               Value of ICNTL(2) is given by INFO(2).
C         -2 -  NC.LE.0. Value of NC is given by INFO(2).
C         -3 -  Error in NR. Value of NR is given by INFO(2).
C         -4 -  NE.LE.0. Value of NE is given by INFO(2).
C         -5 -  LJCN too small. Min. value of LJCN is given by INFO(2).
C         -6 -  LA too small. Min. value of LA is given by INFO(2).
C         -7 -  LIW too small. Value of LIW is given by INFO(2).
C         -8 -  LIP too small. Value of LIP is given by INFO(2).
C         -9 -  The entries of IP not monotonic increasing.
C        -10 -  For each I, IRN(I) or JCN(I) out-of-range.
C        -11 -  ICNTL(6) is out of range.
C         +1 -  One or more duplicated entries. One copy of
C               each such entry is kept and, if ICNTL(3) = 0 and
C               ICNTL(6).GE.0, the values of these entries are
C               added together. If  ICNTL(3) = 0 and ICNTL(6).LT.0,
C               the value of the first occurrence of the entry is used.
C               Initially INFO(3) is set to zero. If an entry appears
C               k times, INFO(3) is incremented by k-1 and INFO(6)
C               is set to the revised number of entries in the
C               matrix.
C         +2 - One or more of the entries in IRN out-of-range. These
C               entries are removed by the routine.`INFO(4) is set to
C               the number of entries which were out-of-range and
C               INFO(6) is set to the revised number of entries in the
C               matrix.
C         +4 - One or more of the entries in JCN out-of-range. These
C               entries are removed by the routine. INFO(5) is set to
C               the number of entries which were out-of-range and
C               INFO(6) is set to the revised number of entries in the
C               matrix. Positive values of INFO(1) are summed so that
C               the user can identify all warnings.
C
C     .. Scalar Arguments ..
      INTEGER LA,LIP,LIW,LJCN,NC,NE,NR
C     ..
C     .. Array Arguments ..
      DOUBLE PRECISION A(LA)
      INTEGER ICNTL(10),IP(LIP),INFO(10),IRN(NE),IW(LIW),JCN(LJCN)
C     ..
C     .. Local Scalars ..
      INTEGER I,ICNTL1,ICNTL2,ICNTL3,ICNTL6,LAA
      INTEGER IDUP,IOUT,IUP,JOUT,LP,MP,KNE,PART
      LOGICAL LCHECK
C     ..
C     .. External Subroutines ..
      EXTERNAL MC59BD,MC59CD,MC59DD,MC59ED,MC59FD
C     ..
C     .. Intrinsic Functions ..
      INTRINSIC MAX
C     ..
C     .. Executable Statements ..

C Initialise
      DO 10 I = 1,10
         INFO(I) = 0
   10 CONTINUE

      ICNTL1 = ICNTL(1)
      ICNTL2 = ICNTL(2)
      ICNTL3 = ICNTL(3)
      ICNTL6 = ICNTL(6)
      LCHECK = (ICNTL1.EQ.0)
C Streams for errors/warnings
      LP = ICNTL(4)
      MP = ICNTL(5)

C  Check the input data
      IF (ICNTL2.GT.2 .OR. ICNTL2.LT.0) THEN
         INFO(1) = -1
         INFO(2) = ICNTL2
         IF (LP.GT.0) THEN
            WRITE (LP,FMT=9000) INFO(1)
            WRITE (LP,FMT=9010) ICNTL2
         END IF
         GO TO 70
      END IF

      IF (ICNTL6.GT.2 .OR. ICNTL6.LT.-2) THEN
         INFO(1) = -11
         INFO(2) = ICNTL6
         IF (LP.GT.0) THEN
            WRITE (LP,FMT=9000) INFO(1)
            WRITE (LP,FMT=9150) ICNTL6
         END IF
         GO TO 70
      END IF
C For real matrices, symmetric = Hermitian so only
C have to distinguish between unsymmetric (ICNTL6 = 0) and
C symmetric (ICNTL6.ne.0)

      IF (NC.LT.1) THEN
        INFO(1) = -2
        INFO(2) = NC
        IF (LP.GT.0) THEN
          WRITE (LP,FMT=9000) INFO(1)
          WRITE (LP,FMT=9020) NC
        END IF
        GO TO 70
      END IF

      IF (NR.LT.1) THEN
        INFO(1) = -3
        INFO(2) = NR
        IF (LP.GT.0) THEN
          WRITE (LP,FMT=9000) INFO(1)
          WRITE (LP,FMT=9030) NR
        END IF
        GO TO 70
      END IF

      IF (ICNTL6.NE.0 .AND. NR.NE.NC) THEN
        INFO(1) = -3
        INFO(2) = NR
        IF (LP.GT.0) THEN
          WRITE (LP,FMT=9000) INFO(1)
          WRITE (LP,FMT=9035) NC,NR
        END IF
        GO TO 70
      END IF

      IF (NE.LT.1) THEN
        INFO(1) = -4
        INFO(2) = NE
        IF (LP.GT.0) THEN
          WRITE (LP,FMT=9000) INFO(1)
          WRITE (LP,FMT=9040) NE
        END IF
        GO TO 70
      END IF

      IF (ICNTL2.EQ.0 .OR. ICNTL2.EQ.1) THEN
        IF (LJCN.LT.NE) THEN
          INFO(1) = -5
          INFO(2) = NE
        END IF
      ELSE
        IF (LJCN.LT.1) THEN
          INFO(1) = -5
          INFO(2) = 1
        END IF
      END IF
      IF (INFO(1).EQ.-5) THEN
         IF (LP.GT.0) THEN
            WRITE (LP,FMT=9000) INFO(1)
            WRITE (LP,FMT=9050) LJCN,INFO(2)
         END IF
         GO TO 70
      END IF

      IF (ICNTL3.EQ.0) THEN
        IF (LA.LT.NE) THEN
          INFO(1) = -6
          INFO(2) = NE
        END IF
      ELSE
        IF (LA.LT.1) THEN
          INFO(1) = -6
          INFO(2) = 1
        END IF
      END IF
      IF (INFO(1).EQ.-6) THEN
         IF (LP.GT.0) THEN
            WRITE (LP,FMT=9000) INFO(1)
            WRITE (LP,FMT=9060) LA,INFO(2)
         END IF
         GO TO 70
      END IF

      IF (ICNTL2.EQ.0 .OR. ICNTL2.EQ.2) THEN
        IF (LIP.LT.NC+1) THEN
          INFO(1) = -7
          INFO(2) = NC+1
        END IF
      ELSE IF (LIP.LT.MAX(NR,NC)+1) THEN
        INFO(1) = -7
        INFO(2) = MAX(NR,NC)+1
      END IF
      IF (INFO(1).EQ.-7) THEN
        IF (LP.GT.0) THEN
          WRITE (LP,FMT=9000) INFO(1)
          WRITE (LP,FMT=9065) LIP,INFO(2)
        END IF
        GO TO 70
      END IF

C Check workspace sufficient
      IF (LIW.LT.MAX(NR,NC)+1) THEN
        INFO(1) = -8
        INFO(2) = MAX(NR,NC)+1
        IF (LP.GT.0) THEN
          WRITE (LP,FMT=9000) INFO(1)
          WRITE (LP,FMT=9070) LIW,INFO(2)
        END IF
        GO TO 70
      END IF

      LAA = NE
      IF (ICNTL3.NE.0) LAA = 1
C Initialise counts of number of out-of-range entries and duplicates
      IOUT = 0
      JOUT = 0
      IDUP = 0
      IUP = 0

C PART is used by MC59BD to indicate if upper or lower or
C all of matrix is required.
C PART =  0 : unsymmetric case, whole matrix wanted
C PART =  1 : symmetric case, lower triangular part of matrix wanted
C PART = -1 : symmetric case, upper triangular part of matrix wanted
      PART = 0
      IF (ICNTL6.NE.0) PART = 1

      IF (ICNTL2.EQ.0) THEN

C Order directly by columns
C On exit from MC59BD, KNE holds number of entries in matrix
C after removal of out-of-range entries. If no data checking, KNE = NE.
        CALL MC59BD(LCHECK,PART,NC,NR,NE,IRN,JCN,LAA,A,IP,IW,
     +              IOUT,JOUT,KNE)
C Return if ALL entries out-of-range.
        IF (KNE.EQ.0) GO TO 50

C Check for duplicates
        IF (LCHECK) CALL MC59ED(NC,NR,NE,IRN,LIP,IP,LAA,A,IW,IDUP,
     &                          KNE,ICNTL6)

      ELSE IF (ICNTL2.EQ.1) THEN

C First order by rows.
C Interchanged roles of IRN and JCN, so set PART = -1
C if matrix is symmetric case
        IF (ICNTL6.NE.0) PART = -1
        CALL MC59BD(LCHECK,PART,NR,NC,NE,JCN,IRN,LAA,A,IW,IP,
     +              JOUT,IOUT,KNE)
C Return if ALL entries out-of-range.
        IF (KNE.EQ.0) GO TO 50

C At this point, JCN and IW hold column indices and row pointers
C Optionally, check for duplicates.
        IF (LCHECK) CALL MC59ED(NR,NC,NE,JCN,NR+1,IW,LAA,A,IP,
     &                          IDUP,KNE,ICNTL6)

C Now order by columns and by rows within each column
        CALL MC59CD(NC,NR,KNE,IRN,JCN,LAA,A,IP,IW)

      ELSE IF (ICNTL2.EQ.2) THEN
C Input is using IP, IRN.
C Optionally check for duplicates and remove out-of-range entries
        IF (LCHECK) THEN
          CALL MC59FD(NC,NR,NE,IRN,NC+1,IP,LAA,A,LIW,IW,IDUP,
     +                IOUT,IUP,KNE,ICNTL6,INFO)
C Return if IP not monotonic.
          IF (INFO(1).EQ.-9) GO TO 40
C Return if ALL entries out-of-range.
          IF (KNE.EQ.0) GO TO 50
        ELSE
           KNE = NE
        END IF

C  Order by rows within each column
        CALL MC59DD(NC,KNE,IRN,IP,LAA,A)

      END IF

      INFO(3) = IDUP
      INFO(4) = IOUT
      INFO(5) = JOUT
      INFO(6) = KNE
      INFO(7) = IUP

C Set warning flag if out-of-range /duplicates found
      IF (IDUP.GT.0) INFO(1) = INFO(1) + 1
      IF (IOUT.GT.0) INFO(1) = INFO(1) + 2
      IF (JOUT.GT.0) INFO(1) = INFO(1) + 4
      IF (INFO(1).GT.0 .AND. MP.GT.0) THEN
        WRITE (MP,FMT=9080) INFO(1)
        IF (IOUT.GT.0) WRITE (MP,FMT=9090) IOUT
        IF (JOUT.GT.0) WRITE (MP,FMT=9110) JOUT
        IF (IDUP.GT.0) WRITE (MP,FMT=9100) IDUP
        IF (IUP.GT.0)  WRITE (MP,FMT=9130) IUP
      END IF
      GO TO 70

   40 INFO(3) = IDUP
      INFO(4) = IOUT
      INFO(7) = IUP
      IF (LP.GT.0) THEN
        WRITE (LP,FMT=9000) INFO(1)
        WRITE (LP,FMT=9140)
      END IF
      GO TO 70

   50 INFO(1) = -10
      INFO(4) = IOUT
      INFO(5) = JOUT
      INFO(2) = IOUT + JOUT
      IF (LP.GT.0) THEN
        WRITE (LP,FMT=9000) INFO(1)
        WRITE (LP,FMT=9120)
      END IF
  70 	RETURN

 9000 FORMAT (/,' *** Error return from MC59AD *** INFO(1) = ',I3)
 9010 FORMAT (1X,'ICNTL(2) = ',I2,' is out of range')
 9020 FORMAT (1X,'NC = ',I6,' is out of range')
 9030 FORMAT (1X,'NR = ',I6,' is out of range')
 9035 FORMAT (1X,'Symmetric case. NC = ',I6,' but NR = ',I6)
 9040 FORMAT (1X,'NE = ',I10,' is out of range')
 9050 FORMAT (1X,'Increase LJCN from ',I10,' to at least ',I10)
 9060 FORMAT (1X,'Increase LA from ',I10,' to at least ',I10)
 9065 FORMAT (1X,'Increase LIP from ',I8,' to at least ',I10)
 9070 FORMAT (1X,'Increase LIW from ',I8,' to at least ',I10)
 9080 FORMAT (/,' *** Warning message from MC59AD *** INFO(1) = ',I3)
 9090 FORMAT (1X,I8,' entries in IRN supplied by the user were ',
     +       /,'       out of range and were ignored by the routine')
 9100 FORMAT (1X,I8,' duplicate entries were supplied by the user')
 9110 FORMAT (1X,I8,' entries in JCN supplied by the user were ',
     +       /,'       out of range and were ignored by the routine')
 9120 FORMAT (1X,'All entries out of range')
 9130 FORMAT (1X,I8,' of these entries were in the upper triangular ',
     +       /,'       part of matrix')
 9140 FORMAT (1X,'Entries in IP are not monotonic increasing')
 9150 FORMAT (1X,'ICNTL(6) = ',I2,' is out of range')
      END
C***********************************************************************
      SUBROUTINE MC59BD(LCHECK,PART,NC,NR,NE,IRN,JCN,LA,A,IP,IW,IOUT,
     +                  JOUT,KNE)
C
C   To sort a sparse matrix from arbitrary order to
C   column order, unordered within each column. Optionally
C   checks for out-of-range entries in IRN,JCN.
C
C LCHECK - logical variable. Intent(IN). If true, check
C          for out-of-range indices.
C PART -   integer variable. Intent(IN)
C PART =  0 : unsymmetric case, whole matrix wanted
C PART =  1 : symmetric case, lower triangular part of matrix wanted
C             (ie IRN(K) .ge. JCN(K) on exit)
C PART = -1 : symmetric case, upper triangular part of matrix wanted
C             (ie IRN(K) .le. JCN(K) on exit)
C   NC - integer variable. Intent(IN)
C      - on entry must be set to the number of columns in the matrix
C   NR - integer variable. Intent(IN)
C      - on entry must be set to the number of rows in the matrix
C   NE - integer variable. Intent(IN)
C      - on entry, must be set to the number of nonzeros in the matrix
C  IRN - integer array of length NE. Intent(INOUT)
C      - on entry set to contain the row indices of the nonzeros
C        in arbitrary order.
C      - on exit, the entries in IRN are reordered so that the row
C        indices for column 1 precede those for column 2 and so on,
C        but the order within columns is arbitrary.
C  JCN - integer array of length NE. Intent(INOUT)
C      - on entry set to contain the column indices of the nonzeros
C      - JCN(K) must be the column index of
C        the entry in IRN(K)
C      - on exit, JCN(K) is the column index for the entry with
C        row index IRN(K) (K=1,...,NE).
C  LA  - integer variable which defines the length of the array A.
C        Intent(IN)
C   A  - real (double precision/complex/complex*16) array of length LA
C        Intent(INOUT)
C      - if LA > 1, the array must be of length NE, and A(K)
C        must be set to the value of the entry in (IRN(K), JCN(K));
C        on exit A is reordered in the same way as IRN
C      - if LA = 1, the array is not accessed
C  IP  - integer array of length NC+1. Intent(INOUT)
C      - not set on entry
C      - on exit, IP(J) contains the position in IRN (and A) of the
C        first entry in column J (J=1,...,NC)
C      - IP(NC+1) is set to NE+1
C  IW  - integer array of length NC+1.  Intent(INOUT)
C      - the array is used as workspace
C      - on exit IW(I) = IP(I) (so IW(I) points to the beginning
C        of column I).
C IOUT - integer variable. Intent(OUT). On exit, holds number
C        of entries in IRN found to be out-of-range
C JOUT - integer variable. Intent(OUT). On exit, holds number
C        of entries in JCN found to be out-of-range
C  KNE - integer variable. Intent(OUT). On exit, holds number
C        of entries in matrix after removal of out-of-range entries.
C        If no data checking, KNE = NE.
C
C    .. Scalar Arguments ..
      INTEGER LA,NC,NE,NR,IOUT,JOUT,KNE,PART
      LOGICAL LCHECK
C    ..
C    .. Array Arguments ..
      DOUBLE PRECISION A(LA)
      INTEGER IP(NC+1),IRN(NE),IW(NC+1),JCN(NE)
C     ..
C     .. Local Scalars ..
      DOUBLE PRECISION ACE,ACEP
      INTEGER I,ICE,ICEP,J,JCE,JCEP,K,L,LOC
C     ..
C     .. Executable Statements ..

C Initialise IW
      DO 10 J = 1,NC + 1
        IW(J) = 0
   10 CONTINUE

      KNE = 0
      IOUT = 0
      JOUT = 0
C ount the number of entries in each column and store in IW.
C We also allow checks for out-of-range indices
      IF (LCHECK) THEN
C heck data.
C Treat case of pattern only separately.
        IF (LA.GT.1) THEN
          IF (PART.EQ.0) THEN
C Unsymmetric
            DO 20 K = 1,NE
              I = IRN(K)
              J = JCN(K)
              IF (I.GT.NR .OR. I.LT.1) THEN
                IOUT = IOUT + 1
C IRN out-of-range. Is JCN also out-of-range?
                IF (J.GT.NC .OR. J.LT.1)  JOUT = JOUT + 1
              ELSE IF (J.GT.NC .OR. J.LT.1) THEN
                JOUT = JOUT + 1
              ELSE
                KNE = KNE + 1
                IRN(KNE) = I
                JCN(KNE) = J
                A(KNE) = A(K)
                IW(J) = IW(J) + 1
              END IF
   20       CONTINUE
          ELSE IF (PART.EQ.1) THEN
C Symmetric, lower triangle
            DO 21 K = 1,NE
              I = IRN(K)
              J = JCN(K)
              IF (I.GT.NR .OR. I.LT.1) THEN
                IOUT = IOUT + 1
C IRN out-of-range. Is JCN also out-of-range?
                IF (J.GT.NC .OR. J.LT.1)  JOUT = JOUT + 1
              ELSE IF (J.GT.NC .OR. J.LT.1) THEN
                JOUT = JOUT + 1
              ELSE
                KNE = KNE + 1
C Lower triangle ... swap if necessary
                IF (I.LT.J) THEN
                  IRN(KNE) = J
                  JCN(KNE) = I
                  IW(I) = IW(I) + 1
                ELSE
                  IRN(KNE) = I
                  JCN(KNE) = J
                  IW(J) = IW(J) + 1
                END IF
                A(KNE) = A(K)
              END IF
   21       CONTINUE
C           ELSE IF (PART.EQ.-1) THEN
C Symmetric, upper triangle
            DO 22 K = 1,NE
              I = IRN(K)
              J = JCN(K)
              IF (I.GT.NR .OR. I.LT.1) THEN
                IOUT = IOUT + 1
C IRN out-of-range. Is JCN also out-of-range?
                IF (J.GT.NC .OR. J.LT.1)  JOUT = JOUT + 1
              ELSE IF (J.GT.NC .OR. J.LT.1) THEN
                JOUT = JOUT + 1
              ELSE
                KNE = KNE + 1
C Upper triangle ... swap if necessary
                IF (I.GT.J) THEN
                  IRN(KNE) = J
                  JCN(KNE) = I
                  IW(I) = IW(I) + 1
                ELSE
                  IRN(KNE) = I
                  JCN(KNE) = J
                  IW(J) = IW(J) + 1
                END IF
                A(KNE) = A(K)
              END IF
   22       CONTINUE
          END IF
        ELSE
C Pattern only
          IF (PART.EQ.0) THEN
            DO 25 K = 1,NE
              I = IRN(K)
              J = JCN(K)
              IF (I.GT.NR .OR. I.LT.1) THEN
                IOUT = IOUT + 1
                IF (J.GT.NC .OR. J.LT.1)  JOUT = JOUT + 1
              ELSE IF (J.GT.NC .OR. J.LT.1) THEN
                JOUT = JOUT + 1
              ELSE
                KNE = KNE + 1
                IRN(KNE) = I
                JCN(KNE) = J
                IW(J) = IW(J) + 1
              END IF
   25       CONTINUE
          ELSE IF (PART.EQ.1) THEN
            DO 26 K = 1,NE
              I = IRN(K)
              J = JCN(K)
              IF (I.GT.NR .OR. I.LT.1) THEN
                IOUT = IOUT + 1
                IF (J.GT.NC .OR. J.LT.1)  JOUT = JOUT + 1
              ELSE IF (J.GT.NC .OR. J.LT.1) THEN
                JOUT = JOUT + 1
              ELSE
                KNE = KNE + 1
C Lower triangle ... swap if necessary
                IF (I.LT.J) THEN
                  IRN(KNE) = J
                  JCN(KNE) = I
                  IW(I) = IW(I) + 1
                ELSE
                  IRN(KNE) = I
                  JCN(KNE) = J
                  IW(J) = IW(J) + 1
                END IF
              END IF
   26       CONTINUE
          ELSE IF (PART.EQ.-1) THEN
            DO 27 K = 1,NE
              I = IRN(K)
              J = JCN(K)
              IF (I.GT.NR .OR. I.LT.1) THEN
                IOUT = IOUT + 1
                IF (J.GT.NC .OR. J.LT.1)  JOUT = JOUT + 1
              ELSE IF (J.GT.NC .OR. J.LT.1) THEN
                JOUT = JOUT + 1
              ELSE
                KNE = KNE + 1
C Upper triangle ... swap if necessary
                IF (I.GT.J) THEN
                  IRN(KNE) = J
                  JCN(KNE) = I
                  IW(I) = IW(I) + 1
                ELSE
                  IRN(KNE) = I
                  JCN(KNE) = J
                  IW(J) = IW(J) + 1
                END IF
              END IF
   27       CONTINUE
          END IF
        END IF
C Return if ALL entries out-of-range.
        IF (KNE.EQ.0) GO TO 130

      ELSE

C No checks
        KNE = NE
        IF (PART.EQ.0) THEN
          DO 30 K = 1,NE
            J = JCN(K)
            IW(J) = IW(J) + 1
   30     CONTINUE
        ELSE IF (PART.EQ.1) THEN
          DO 35 K = 1,NE
            I = IRN(K)
            J = JCN(K)
C Lower triangle ... swap if necessary
            IF (I.LT.J) THEN
               IRN(K) = J
               JCN(K) = I
               IW(I) = IW(I) + 1
            ELSE
              IW(J) = IW(J) + 1
            END IF
   35     CONTINUE
        ELSE IF (PART.EQ.-1) THEN
          DO 36 K = 1,NE
            I = IRN(K)
            J = JCN(K)
C Upper triangle ... swap if necessary
            IF (I.GT.J) THEN
               IRN(K) = J
               JCN(K) = I
               IW(I) = IW(I) + 1
            ELSE
              IW(J) = IW(J) + 1
            END IF
   36     CONTINUE
        END IF
      END IF

C KNE is now the number of nonzero entries in matrix.

C Put into IP and IW the positions where each column
C would begin in a compressed collection with the columns
C in natural order.

      IP(1) = 1
      DO 37 J = 2,NC + 1
        IP(J) = IW(J-1) + IP(J-1)
        IW(J-1) = IP(J-1)
   37 CONTINUE

C Reorder the elements into column order.
C Fill in each column from the front, and as a new entry is placed
C in column K increase the pointer IW(K) by one.

      IF (LA.EQ.1) THEN
C Pattern only
        DO 70 L = 1,NC
          DO 60 K = IW(L),IP(L+1) - 1
            ICE = IRN(K)
            JCE = JCN(K)
            DO 40 J = 1,NE
              IF (JCE.EQ.L) GO TO 50
              LOC = IW(JCE)
              JCEP = JCN(LOC)
              ICEP = IRN(LOC)
              IW(JCE) = LOC + 1
              JCN(LOC) = JCE
              IRN(LOC) = ICE
              JCE = JCEP
              ICE = ICEP
   40       CONTINUE
   50       JCN(K) = JCE
            IRN(K) = ICE
   60     CONTINUE
   70   CONTINUE
      ELSE

        DO 120 L = 1,NC
          DO 110 K = IW(L),IP(L+1) - 1
            ICE = IRN(K)
            JCE = JCN(K)
            ACE = A(K)
            DO 90 J = 1,NE
              IF (JCE.EQ.L) GO TO 100
              LOC = IW(JCE) 			  
				
              JCEP = JCN(LOC)
              ICEP = IRN(LOC)
              IW(JCE) = LOC + 1
              JCN(LOC) = JCE
              IRN(LOC) = ICE
              JCE = JCEP
              ICE = ICEP
              ACEP = A(LOC)
              A(LOC) = ACE
              ACE = ACEP
   90       CONTINUE

  100       JCN(K) = JCE
            IRN(K) = ICE
            A(K) = ACE
  110     CONTINUE
  120   CONTINUE
      END IF

  130 CONTINUE

      RETURN
      END
C
C**********************************************************
      SUBROUTINE MC59CD(NC,NR,NE,IRN,JCN,LA,A,IP,IW)
C
C   To sort a sparse matrix stored by rows,
C   unordered within each row, to ordering by columns, with
C   ordering by rows within each column.
C
C   NC - integer variable. Intent(IN)
C      - on entry must be set to the number of columns in the matrix
C   NR - integer variable. Intent(IN)
C      - on entry must be set to the number of rows in the matrix
C  NE - integer variable. Intent(IN)
C      - on entry, must be set to the number of nonzeros in the matrix
C  IRN - integer array of length NE. Intent(OUT).
C      - not set on entry.
C      - on exit,  IRN holds row indices with the row
C        indices for column 1 preceding those for column 2 and so on,
C        with ordering by rows within each column.
C  JCN - integer array of length NE. Intent(INOUT)
C      - on entry set to contain the column indices of the nonzeros
C        with indices for column 1 preceding those for column 2
C        and so on, with the order within columns is arbitrary.
C      - on exit, contents destroyed.
C  LA  - integer variable which defines the length of the array A.
C        Intent(IN)
C   A  - real (double precision/complex/complex*16) array of length LA
C        Intent(INOUT)
C      - if LA > 1, the array must be of length NE, and A(K)
C        must be set to the value of the entry in JCN(K);
C        on exit A, A(K) holds the value of the entry in IRN(K).
C      - if LA = 1, the array is not accessed
C  IP  - integer array of length NC+1. Intent(INOUT)
C      - not set on entry
C      - on exit, IP(J) contains the position in IRN (and A) of the
C        first entry in column J (J=1,...,NC)
C      - IP(NC+1) is set to NE+1
C  IW  - integer array of length NR+1.  Intent(IN)
C      - on entry, must be set on entry so that IW(J) points to the
C        position in JCN of the first entry in row J, J=1,...,NR, and
C        IW(NR+1) must be set to NE+1
C
C     .. Scalar Arguments ..
      INTEGER LA,NC,NE,NR
C     ..
C     .. Array Arguments ..
      DOUBLE PRECISION A(LA)
      INTEGER IP(NC+1),IRN(NE),IW(NR+1),JCN(NE)
C     ..
C     .. Local Scalars ..
      DOUBLE PRECISION ACE,ACEP
      INTEGER I,ICE,ICEP,J,J1,J2,K,L,LOC,LOCP
C     ..
C     .. Executable Statements ..

C  Count the number of entries in each column

      DO 10 J = 1,NC
        IP(J) = 0
   10 CONTINUE

      IF (LA.GT.1) THEN

        DO 20 K = 1,NE
          I = JCN(K)
          IP(I) = IP(I) + 1
          IRN(K) = JCN(K)
   20   CONTINUE
        IP(NC+1) = NE + 1

C  Set IP so that IP(I) points to the first entry in column I+1

        IP(1) = IP(1) + 1
        DO 30 J = 2,NC
          IP(J) = IP(J) + IP(J-1)
   30   CONTINUE

        DO 50 I = NR,1,-1
          J1 = IW(I)
          J2 = IW(I+1) - 1
          DO 40 J = J1,J2
            K = IRN(J)
            L = IP(K) - 1
            JCN(J) = L
            IRN(J) = I
            IP(K) = L
   40     CONTINUE
   50   CONTINUE
        IP(NC+1) = NE + 1
        DO 70 J = 1,NE
          LOC = JCN(J)
          IF (LOC.EQ.0) GO TO 70
          ICE = IRN(J)
          ACE = A(J)
          JCN(J) = 0
          DO 60 K = 1,NE
            LOCP = JCN(LOC)
            ICEP = IRN(LOC)
            ACEP = A(LOC)
            JCN(LOC) = 0
            IRN(LOC) = ICE
            A(LOC) = ACE
            IF (LOCP.EQ.0) GO TO 70
            ICE = ICEP
            ACE = ACEP
            LOC = LOCP
   60     CONTINUE
   70   CONTINUE
      ELSE

C Pattern only

C  Count the number of entries in each column

        DO 90 K = 1,NE
          I = JCN(K)
          IP(I) = IP(I) + 1
   90   CONTINUE
        IP(NC+1) = NE + 1

C  Set IP so that IP(I) points to the first entry in column I+1

        IP(1) = IP(1) + 1
        DO 100 J = 2,NC
          IP(J) = IP(J) + IP(J-1)
  100   CONTINUE

        DO 120 I = NR,1,-1
          J1 = IW(I)
          J2 = IW(I+1) - 1
          DO 110 J = J1,J2
            K = JCN(J)
            L = IP(K) - 1
            IRN(L) = I
            IP(K) = L
  110     CONTINUE
  120   CONTINUE

      END IF

      RETURN
      END

C**********************************************************

      SUBROUTINE MC59DD(NC,NE,IRN,IP,LA,A)
C
C To sort from arbitrary order within each column to order
C by increasing row index. Note: this is taken from MC20B/BD.
C
C   NC - integer variable. Intent(IN)
C      - on entry must be set to the number of columns in the matrix
C   NE - integer variable. Intent(IN)
C      - on entry, must be set to the number of nonzeros in the matrix
C  IRN - integer array of length NE. Intent(INOUT)
C      - on entry set to contain the row indices of the nonzeros
C        ordered so that the row
C        indices for column 1 precede those for column 2 and so on,
C        but the order within columns is arbitrary.
C        On exit, the order within each column is by increasing
C        row indices.
C   LA - integer variable which defines the length of the array A.
C        Intent(IN)
C    A - real (double precision/complex/complex*16) array of length LA
C        Intent(INOUT)
C      - if LA > 1, the array must be of length NE, and A(K)
C        must be set to the value of the entry in IRN(K);
C        on exit A is reordered in the same way as IRN
C      - if LA = 1, the array is not accessed
C  IP  - integer array of length NC. Intent(IN)
C      - on entry, IP(J) contains the position in IRN (and A) of the
C        first entry in column J (J=1,...,NC)
C     . .
C     .. Scalar Arguments ..
      INTEGER LA,NC,NE
C     ..
C     .. Array Arguments ..
      DOUBLE PRECISION A(LA)
      INTEGER IRN(NE),IP(NC)
C     ..
C     .. Local Scalars ..
      DOUBLE PRECISION ACE
      INTEGER ICE,IK,J,JJ,K,KDUMMY,KLO,KMAX,KOR
C     ..
C     .. Intrinsic Functions ..
      INTRINSIC ABS
C     ..
C     .. Executable Statements ..

C Jump if pattern only.
      IF (LA.GT.1) THEN
        KMAX = NE
        DO 50 JJ = 1,NC
          J = NC + 1 - JJ
          KLO = IP(J) + 1
          IF (KLO.GT.KMAX) GO TO 40
          KOR = KMAX
          DO 30 KDUMMY = KLO,KMAX
C Items KOR, KOR+1, .... ,KMAX are in order
            ACE = A(KOR-1)
            ICE = IRN(KOR-1)
            DO 10 K = KOR,KMAX
              IK = IRN(K)
              IF (ABS(ICE).LE.ABS(IK)) GO TO 20
              IRN(K-1) = IK
              A(K-1) = A(K)
   10       CONTINUE
            K = KMAX + 1
   20       IRN(K-1) = ICE
            A(K-1) = ACE
            KOR = KOR - 1
   30     CONTINUE
C Next column
   40     KMAX = KLO - 2
   50   CONTINUE
      ELSE

C Pattern only.
        KMAX = NE
        DO 150 JJ = 1,NC
          J = NC + 1 - JJ
          KLO = IP(J) + 1
          IF (KLO.GT.KMAX) GO TO 140
          KOR = KMAX
          DO 130 KDUMMY = KLO,KMAX
C Items KOR, KOR+1, .... ,KMAX are in order
            ICE = IRN(KOR-1)
            DO 110 K = KOR,KMAX
              IK = IRN(K)
              IF (ABS(ICE).LE.ABS(IK)) GO TO 120
              IRN(K-1) = IK
  110       CONTINUE
            K = KMAX + 1
  120       IRN(K-1) = ICE
            KOR = KOR - 1
  130     CONTINUE
C Next column
  140     KMAX = KLO - 2
  150   CONTINUE
      END IF
      END
C***********************************************************************

      SUBROUTINE MC59ED(NC,NR,NE,IRN,LIP,IP,LA,A,IW,IDUP,KNE,ICNTL6)

C Checks IRN for duplicate entries.
C On exit, IDUP holds number of duplicates found and KNE is number
C of entries in matrix after removal of duplicates
C     . .
C     .. Scalar Arguments ..
      INTEGER ICNTL6,IDUP,KNE,LIP,LA,NC,NR,NE
C     ..
C     .. Array Arguments ..
      DOUBLE PRECISION A(LA)
      INTEGER IRN(NE),IP(LIP),IW(NR)
C     ..
C     .. Local Scalars ..
      INTEGER I,J,K,KSTART,KSTOP,NZJ

      IDUP = 0
      KNE = 0
C Initialise IW
      DO 10 I = 1,NR
        IW(I) = 0
   10 CONTINUE

      KSTART = IP(1)
      IF (LA.GT.1) THEN
C Matrix entries considered
        NZJ = 0
        DO 30 J = 1,NC
          KSTOP = IP(J+1)
          IP(J+1) = IP(J)
          DO 20 K = KSTART,KSTOP - 1
            I = IRN(K)
            IF (IW(I).LE.NZJ) THEN
              KNE = KNE + 1
              IRN(KNE) = I
              A(KNE) = A(K)
              IP(J+1) = IP(J+1) + 1
              IW(I) = KNE
            ELSE
C We have a duplicate in column J
              IDUP = IDUP + 1
C If requested, sum duplicates
              IF (ICNTL6.GE.0) A(IW(I)) = A(IW(I)) + A(K)
            END IF
   20     CONTINUE
          KSTART = KSTOP
          NZJ = KNE
   30   CONTINUE

      ELSE

C Pattern only
        DO 50 J = 1,NC
          KSTOP = IP(J+1)
          IP(J+1) = IP(J)
          DO 40 K = KSTART,KSTOP - 1
            I = IRN(K)
            IF (IW(I).LT.J) THEN
              KNE = KNE + 1
              IRN(KNE) = I
              IP(J+1) = IP(J+1) + 1
              IW(I) = J
            ELSE
C  We have a duplicate in column J
              IDUP = IDUP + 1
            END IF
   40     CONTINUE
          KSTART = KSTOP
   50   CONTINUE
      END IF

      RETURN
      END
C***********************************************************************

      SUBROUTINE MC59FD(NC,NR,NE,IRN,LIP,IP,LA,A,LIW,IW,IDUP,IOUT,
     +                  IUP,KNE,ICNTL6,INFO)

C Checks IRN for duplicate and out-of-range entries.
C For symmetric matrix, also checks NO entries lie in upper triangle.
C Also checks IP is monotonic.
C On exit:
C IDUP holds number of duplicates found
C IOUT holds number of out-of-range entries
C For symmetric matrix, IUP holds number of entries in upper
C triangular part.
C KNE holds number of entries in matrix after removal of
C out-of-range and duplicate entries.
C Note: this is similar to MC59ED except it also checks IP is
C monotonic and removes out-of-range entries in IRN.
C     . .
C     .. Scalar Arguments ..
      INTEGER ICNTL6,IDUP,IOUT,IUP,KNE,LA,LIP,LIW,NC,NR,NE
C     ..
C     .. Array Arguments ..
      DOUBLE PRECISION A(LA)
      INTEGER IRN(NE),IP(LIP),IW(LIW),INFO(2)
C     ..
C     .. Local Scalars ..
      INTEGER I,J,K,KSTART,KSTOP,NZJ,LOWER

      IDUP = 0
      IOUT = 0
      IUP = 0
      KNE = 0
C Initialise IW
      DO 10 I = 1,NR
        IW(I) = 0
   10 CONTINUE

      KSTART = IP(1)
      LOWER = 1
      IF (LA.GT.1) THEN
        NZJ = 0
        DO 30 J = 1,NC
C In symmetric case, entries out-of-range if they lie
C in upper triangular part.
          IF (ICNTL6.NE.0) LOWER = J
          KSTOP = IP(J+1)
          IF (KSTART.GT.KSTOP) THEN
            INFO(1) = -9
            INFO(2) = J
            RETURN
          END IF
          IP(J+1) = IP(J)
          DO 20 K = KSTART,KSTOP - 1
            I = IRN(K)
C Check for out-of-range
            IF (I.GT.NR .OR. I.LT.LOWER) THEN
              IOUT = IOUT + 1
C In symmetric case, check if entry is out-of-range because
C it lies in upper triangular part.
              IF (ICNTL6.NE.0 .AND. I.LT.J) IUP = IUP + 1
            ELSE IF (IW(I).LE.NZJ) THEN
              KNE = KNE + 1
              IRN(KNE) = I
              A(KNE) = A(K)
              IP(J+1) = IP(J+1) + 1
              IW(I) = KNE
            ELSE
C  We have a duplicate in column J
              IDUP = IDUP + 1
C If requested, sum duplicates
              IF (ICNTL6.GE.0) A(IW(I)) = A(IW(I)) + A(K)
            END IF
   20     CONTINUE
          KSTART = KSTOP
          NZJ = KNE
   30   CONTINUE

      ELSE

C Pattern only
        DO 50 J = 1,NC
C In symmetric case, entries out-of-range if lie
C in upper triangular part.
          IF (ICNTL6.NE.0) LOWER = J
          KSTOP = IP(J+1)
          IF (KSTART.GT.KSTOP) THEN
            INFO(1) = -9
            INFO(2) = J
            RETURN
          END IF
          IP(J+1) = IP(J)
          DO  40 K = KSTART,KSTOP - 1
            I = IRN(K)
C Check for out-of-range
            IF (I.GT.NR .OR. I.LT.LOWER) THEN
              IOUT = IOUT + 1
              IF (ICNTL6.NE.0 .AND. I.GT.1) IUP = IUP + 1
            ELSE IF (IW(I).LT.J) THEN
              KNE = KNE + 1
              IRN(KNE) = I
              IP(J+1) = IP(J+1) + 1
              IW(I) = J
            ELSE
C  We have a duplicate in column J
              IDUP = IDUP + 1
            END IF
   40     CONTINUE
          KSTART = KSTOP
   50   CONTINUE
      END IF

      RETURN
      END
