 & "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe" naive_multiply_kernel_optimize_2.cu -o naive_multiply_kernel_optimize_2.exe -ccbin "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64" -std=c++14 -Xcompiler "/utf-8 /EHsc /Od /Zi /MD" -Xptxas -O0 -G -allow-unsupported-compiler 



  & "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe" naive_multiply_kernel_0.cu -o naive_0.exe -ccbin "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64" -std=c++14 -Xcompiler "/utf-8 /EHsc /Od /Zi /MD" -Xptxas -O0 -G -allow-unsupported-compiler 



  & "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe" naive_multiply_kernel_optimize_1.cu -o naive_1.exe -ccbin "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64" -std=c++14 -Xcompiler "/utf-8 /EHsc /Od /Zi /MD" -Xptxas -O0 -G -allow-unsupported-compiler 


   & "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe" naive_multiply_kernel_optimize_2.cu -o naive_2.exe -ccbin "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64" -std=c++14 -Xcompiler "/utf-8 /EHsc /Od /Zi /MD" -Xptxas -O0 -G -allow-unsupported-compiler 



   & "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe" win_naive_multiply_kernel_0.cu -o naive_win_0.exe -ccbin "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64" -std=c++14 -Xcompiler "/utf-8 /EHsc /Od /Zi /MD" -Xptxas -O0 -G -allow-unsupported-compiler 


    & "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe" win_naive_multiply_kernel_1.cu -o naive_win_1.exe -ccbin "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64" -std=c++14 -Xcompiler "/utf-8 /EHsc /Od /Zi /MD" -Xptxas -O0 -G -allow-unsupported-compiler 

     & "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe" win_naive_multiply_kernel_2.cu -o naive_win_2.exe -ccbin "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64" -std=c++14 -Xcompiler "/utf-8 /EHsc /Od /Zi /MD" -Xptxas -O0 -G -allow-unsupported-compiler 

     & "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe" win_naive_multiply_kernel_3.cu -o naive_win_3.exe -ccbin "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64" -std=c++14 -Xcompiler "/utf-8 /EHsc /O2 /MD" -allow-unsupported-compiler 


      & "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe" win_naive_multiply_kernel_3.cu -o naive_win_3_release.exe -ccbin "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64" -std=c++14 -Xcompiler "/utf-8 /EHsc /O2 /MD" -Xptxas -O3  -allow-unsupported-compiler


       & "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe" win_naive_multiply_kernel_4.cu -o naive_win_4.exe -ccbin "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64" -std=c++14 -Xcompiler "/utf-8 /EHsc /O2 /MD" -allow-unsupported-compiler


        & "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe" win_naive_multiply_kernel_4.cu -o naive_win_4_release.exe -ccbin "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64" -std=c++14 -Xcompiler "/utf-8 /EHsc /O2 /MD" -Xptxas -O3  -allow-unsupported-compiler


         & "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe" win_naive_multiply_kernel_5.cu -o naive_win_5.exe -ccbin "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64" -std=c++14 -Xcompiler "/utf-8 /EHsc /O2 /MD" -allow-unsupported-compiler


          & "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe" win_naive_multiply_kernel_3_dot_5.cu -o naive_win_3_dot_5.exe -ccbin "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64" -std=c++14 -Xcompiler "/utf-8 /EHsc /O2 /MD" -allow-unsupported-compiler


          & "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe" win_naive_multiply_kernel_2.cu -o naive_win_2_release.exe -ccbin "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64" -std=c++14 -Xcompiler "/utf-8 /EHsc /O2 /MD" -Xptxas -O3  -allow-unsupported-compiler


           # CuTe（CUTLASS）：头文件在 CUTLASS 仓库的 include 目录；按需改 -I 路径与 -arch
           & "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe" win_cute_gemm.cu -o win_cute_gemm.exe -I"C:\Users\primelee\Desktop\cutlass\include" -std=c++17 -ccbin "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64" -Xcompiler "/utf-8 /EHsc /O2 /MD /Zc:preprocessor" -allow-unsupported-compiler -arch=native


            # CuTe GEMM v2：面向 Turing / RTX 2060 Max-Q，固定 -arch=sm_75
            & "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe" win_cute_gemm_2.cu -o win_cute_gemm_2.exe -I"C:\Users\primelee\Desktop\cutlass\include" -std=c++17 -arch=sm_75 -ccbin "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64" -Xcompiler "/utf-8 /EHsc /O2 /MD /Zc:preprocessor" -allow-unsupported-compiler -Xptxas -O3