%
%       o - - - - - - - - - - - - - - - - - - - - - - - o
%       |                                               |
%       |     MATH107 Lab - Final Project               |
%       |     Author: Brian Fong                        |
%       |     ID: bfong2,  87658711                     |
%       |     Date: 05/13/2021                          |
%       |                                               |
%       o - - - - - - - - - - - - - - - - - - - - - - - o 
%
%% Reset
clear
close
clc
%
% Consider the heat equation with boundary conditions given by 
%
%   u_t     = 1.5 u_xx + x^3 e^t - 9 x e^t,     (x, t) in (0, 1) X (0, T)
%   IC: u(x, 0) = x^3                               x in [0, 1]
%   BC: u(0, t) = 0,       u_x(1, t) = 3 e^t        t in [0, T]
%
% whose exact solution is u(x, t) = x^3 e^t 

%% ======== PART I =============================================
%   Use Forward Euler's method in time and Finite Difference method in
%   space to approximate the solution to the problem where the Neumann
%   boundary condition is approximated using the first order
%   approximation given by
%       u_x(1, t) ~= (v_m(t)-v_m-1(t)) / h
%   Obtain approximations at T = Nk = 0.5 and determine 
%       max_i |u(ih, Nk)-w_i^N| for h = 1/10, 1/20, 1/40, 1/80, 1/160.
%   Choose k so that k/h^2 = 1/3. Determine the order of accuracy in
%   space. 

% Exact Solution
u = @(x, t) x.^3.*exp(t);   

% Basic Setting
a = 0; b = 1; 
c = 0; T = 0.5; 
alpha = sqrt(1.5); 
H = [1/10, 1/20, 1/40, 1/80, 1/160];

% Plotting Exact Solution 
exact_x = a:10^-5:b;
subplot(1, 2, 1)
plot(exact_x, u(exact_x, T), 'k--')
hold on
sgtitle("Forward Euler's Method")
title("Approximation (at time T = 0.5)")
xlabel("x")
ylabel("u")
x_text = exact_x(round(size(exact_x, 2)/2));
y_text = u(x_text, T);
text(x_text, y_text, "Exact")
axis([0 1 0 2])
x0=100;
y0=75;
width=1000;
height=900;
set(gcf,'position',[x0,y0,width,height])

% Main Algorithm 
errors = zeros(size(H));
for h_i = 1:size(H, 2)
	% Basic Setting
	h = H(h_i);     x = a:h:b; 		Nx = size(x, 2);	Nix = Nx-2;
	k = 1/3*h^2;    t = c:k:T;      Nt = size(t, 2);	
	lambda = alpha^2*k/h^2; 
	
	% Constructing Matrix
	A = diag((1-2*lambda)*ones(Nix, 1))...		
		+ diag(lambda*ones(Nix-1, 1), 1)...		
		+ diag(lambda*ones(Nix-1, 1), -1);		

	% "Nonhomogeneous part"
	P = @(x, t) x.^3*exp(t) - 9*x.*exp(t);

	% Initial Conditions
    w = zeros(Nx, Nt);
	w(:, 1) = x.^3'; 
	
	% Boundary Conditions
	boundary = zeros(size(A, 1), 1); 

	% Forward Euler's Method
    for j = 1:Nt-1
		% update boundary at x = 1
		boundary(end) = lambda*w(end, j);
		% update
		w(2:Nx-1, j+1) = A*w(2:Nx-1, j)...
                        + k*P(x(2:Nx-1), t(j))'...
                        + boundary;
		% apply boundary condition
		w(end, j+1) = w(end-1, j+1) + 3*h*exp(t(j+1));
    end
    % Plotting Results
    plot(x, w(:, end))
    h_str = strcat("h=",num2str(h)); 
    mid_index = round(size(x, 2)/2) + h_i; 
    x_text = x(mid_index);
    y_text = w(mid_index, end);
    text(x_text, y_text, h_str)
    
	% Error
    abs_err = abs(w(2:Nx-1, end) - u(x(2:Nx-1), t(end-1))');
	errors(h_i) = max(abs_err);
end
legend("Exact", "1/10", "1/20", "1/40", "1/80", "1/160", "Location", "northwest")
hold off

% Plotting Error
subplot(1, 2, 2)
loglog(H, H)
hold on
loglog(H, H.^2)
plot(H, errors, '-o')
xlabel("Step Size h")
ylabel("Absolute Error")
title("Max Absolute Error vs Step Size (h)")
legend("O(h)", "O(h^2)", "Forward Euler", "Location", "northwest")
hold off

% Printing Error
fprintf("Forward Euler's Method: \n\n")
fprintf("  Step Size (h)     max_i|u(ih, Nk) - w_i^N| \n")
fprintf("------------------------------------------------\n")
for i = 1:size(H, 2)
   fprintf("    %1.5f                 %10.5f \n", H(i), errors(i)) 
end
fprintf("\n")

% DISCUSSION:
%   Looking at the results of the error behind the Forward Euler's 
%   method, we see that the error is around the same order of 
%   magnitude as the step size. Thus, this method is of order O(h)

%% ======== PART II =============================================
%   Repeat Part (I) using the second order approximation 
%       u_x(1, t) ~= (v_m+1(t) - v_m-1(t))/(2h) 
%   with a ghost point x_m+1. Determine the order of the accuracy 
%   in space. 
clear

% Exact Solution
u = @(x, t) x.^3.*exp(t);   

% Basic Setting
a = 0; b = 1; 
c = 0; T = 0.5; 
alpha = sqrt(1.5); 
H = [1/10, 1/20, 1/40, 1/80, 1/160];

% Plotting Exact Solution 
exact_x = a:10^-5:b;
figure
subplot(1, 2, 1)
plot(exact_x, u(exact_x, T), 'k--')
hold on
sgtitle("FDM with 2nd-Order Approx. (Ghost Point x_{m+1})")
title("Approximation (at time T = 0.5)")
xlabel("x")
ylabel("u")
x_text = exact_x(round(size(exact_x, 2)/2));
y_text = u(x_text, T);
text(x_text, y_text, "Exact")
axis([0 1 0 2])
x0=500;
y0=75;
width=1000;
height=900;
set(gcf,'position',[x0,y0,width,height])

% Main Algorithm 
errors = zeros(size(H));
for h_i = 1:size(H, 2)
	% Basic Setting
	h = H(h_i);     x = a:h:b; 		Nx = size(x, 2);	Nix = Nx-2;
	k = 1/3*h^2;    t = c:k:T;      Nt = size(t, 2);	
	lambda = alpha^2*k/h^2; 
	
	% Constructing Matrix
	A = diag((1-2*lambda)*ones(Nix, 1))...		
		+ diag(lambda*ones(Nix-1, 1), 1)...		
		+ diag(lambda*ones(Nix-1, 1), -1);		

	% "Nonhomogeneous part"
	P = @(x, t) x.^3*exp(t) - 9*x.*exp(t);

	% Initial Conditions
    w = zeros(Nx, Nt);
	w(:, 1) = x.^3'; 
	
	% Boundary Conditions
	boundary = zeros(size(A, 1), 1); 

	% 2nd-Order Approximation with ghost point
    for j = 1:Nt-1
		% update boundary at x = 1
		boundary(end) = lambda*w(end, j);
		% update
		w(2:Nx-1, j+1) = A*w(2:Nx-1, j)...
                        + k*P(x(2:Nx-1), t(j))'...
                        + boundary;
		% apply boundary condition
		w(end, j+1) = (1-2*lambda)*w(end, j)... 
                        + lambda*(6*h*exp(t(j)) + w(end-1, j))... 
                        + lambda*w(end-1, j)...
                        + k*P(x(end), t(j))';
    end
    % Plotting Results
    plot(x, w(:, end))
    h_str = strcat("h=",num2str(h)); 
    mid_index = round(size(x, 2)/2) + h_i; 
    x_text = x(mid_index);
    y_text = w(mid_index, end);
    text(x_text, y_text, h_str)
    
	% Error
    abs_err = abs(w(2:Nx-1, end) - u(x(2:Nx-1), t(end-1))');
	errors(h_i) = max(abs_err);
end
legend("Exact", "1/10", "1/20", "1/40", "1/80", "1/160", "Location", "northwest")
hold off

% Plotting Error
subplot(1, 2, 2)
loglog(H, H)
hold on 
loglog(H, H.^2)
loglog(H, errors, '-o')
xlabel("Step Size h")
ylabel("Absolute Error")
title("Max Absolute Error vs Step Size (h)")
legend("O(h)", "O(h^2)", "FDM", "Location", "northwest")
hold off

% Printing Error
fprintf("FDM with 2nd-Order Approximation (Ghost Point x_{m+1}): \n\n")
fprintf("  Step Size (h)     max_i|u(ih, Nk) - w_i^N| \n")
fprintf("--------------------------------------------------\n")
for i = 1:size(H, 2)
   fprintf("    %1.5f                 %10.5f \n", H(i), errors(i)) 
end
fprintf("\n")

% DISCUSSION:
%   Looking at the results of the error behind the 2nd-Order
%   Approximation with ghost point x_{m+1} method, we see that
%   the error is around O(h^2); more accurate than the
%   Forward Euler method. 

%% ======== PART III ============================================
%   Use the Crank-Nicolson method in time and Finite Difference method
%   in space to repeat Part (I). Choose k = h. Determine the order of
%   accuracy in space. 
clear

% Exact Solution
u = @(x, t) x.^3.*exp(t);   

% Basic Setting
a = 0; b = 1; 
c = 0; T = 0.5; 
alpha = sqrt(1.5); 
H = [1/10, 1/20, 1/40, 1/80, 1/160];

% Plotting Exact Solution 
exact_x = a:10^-5:b;
figure
subplot(1, 2, 1)
plot(exact_x, u(exact_x, T), 'k--')
hold on
sgtitle("Crank-Nicolson Method")
title("Approximation (at time T = 0.5)")
xlabel("x")
ylabel("u")
x_text = exact_x(round(size(exact_x, 2)/2));
y_text = u(x_text, T);
text(x_text, y_text, "Exact")
axis([0 1 0 2])
x0=900;
y0=75;
width=1000;
height=900;
set(gcf,'position',[x0,y0,width,height])

% Main Algorithm 
errors = zeros(size(H));
for h_i = 1:size(H, 2)
	% Basic Setting
	h = H(h_i);     x = a:h:b; 		Nx = size(x, 2);	Nix = Nx-2;
	k = h^2;    t = c:k:T;      Nt = size(t, 2);	
	lambda = alpha^2*k/h^2; 
	
	% Constructing Matrix
    % forward
    A = diag((1-2*lambda)*ones(Nix, 1))...		
		+ diag(lambda*ones(Nix-1, 1), 1)...		
		+ diag(lambda*ones(Nix-1, 1), -1);		
	% backward
    B = diag((1+2*lambda)*ones(Nix, 1))...		
		+ diag(-lambda*ones(Nix-1, 1), 1)...		
		+ diag(-lambda*ones(Nix-1, 1), -1);		

	% "Nonhomogeneous part"
	P = @(x, t) x.^3*exp(t) - 9*x.*exp(t);

	% Initial Conditions
    w_f = zeros(Nx, Nt);
	w_f(:, 1) = x.^3'; 
    w_b = zeros(Nx, Nt);
	w_b(:, 1) = x.^3'; 
	
	% Boundary Conditions
	boundary = zeros(size(A, 1), 1); 
    
    % Forward Euler's Method
    for j = 1:Nt-1
		% update boundary at x = 1
		boundary(end) = lambda/2*w_f(end, j);
		% update
		w_f(2:Nx-1, j+1) = A*w_f(2:Nx-1, j)...
                        + k*P(x(2:Nx-1), t(j))'...
                        + boundary;
		% apply boundary condition
		w_f(end, j+1) = w_f(end-1, j+1) + 3*h*exp(t(j+1));
    end

	% Backward Euler's Method
    for j = 1:Nt-1
		% update boundary at x = 1
		boundary(end) = lambda/2*w_b(end, j);
		% update
        w_b(2:Nx-1, j+1) = B \ (A*w_b(2:Nx-1, j+1) + k*P(x(2:Nx-1), t(j))' + boundary);
		% apply boundary condition
		w_b(end, j+1) = w_b(end-1, j+1) + 3*h*exp(t(j+1));
    end
    
    % Take average of Forward and Backward Euler
    w = 1/2*(w_f + w_b);
    
    % Plotting Results
    plot(x, w(:, end))
    h_str = strcat("h=",num2str(h)); 
    mid_index = round(size(x, 2)/2) + h_i; 
    x_text = x(mid_index);
    y_text = w(mid_index, end);
    text(x_text, y_text, h_str)
    
    % Error
    abs_err = abs(w(2:Nx-1, end) - u(x(2:Nx-1), t(end-1))');
    errors(h_i) = max(abs_err);
end
legend("Exact", "1/10", "1/20", "1/40", "1/80", "1/160", "Location", "northwest")
hold off

% Plotting Error
subplot(1, 2, 2)
loglog(H, H)
hold on
loglog(H, H.^2)
plot(H, errors, '-o')
xlabel("Step Size h")
ylabel("Absolute Error")
title("Max Absolute Error vs Step Size (h)")
legend("O(h)", "O(h^2)", "Forward Euler", "Location", "northwest")
hold off

% Printing Error
fprintf("Forward Euler's Method: \n\n")
fprintf("  Step Size (h)     max_i|u(ih, Nk) - w_i^N| \n")
fprintf("------------------------------------------------\n")
for i = 1:size(H, 2)
   fprintf("    %1.5f                 %5.5f \n", H(i), errors(i)) 
end
fprintf("\n")

% DISCUSSION:
%   Since we are using a Finite-Difference method with 2nd-order 
%   approximation, this method should result in O(h^2) accuracy 
%   in space. 
%   


%% ======== PART IV =============================================
% Discuss the following question 
%   (i) What is the order of accuracy in Part (I)? Why do you think that
%   you obtain this order of accuracy? 
%   
%       The Forward Euler method in space uses the 1st-order Taylor 
%       series in its approximation. This corresponds to the n = 1
%       Taylor method, which we know has order of O(h^n) = O(h).
%       Moreover, this method uses the given Neumann boundary condition
%       which is approximated with an O(h) approximation. 
%       In addition, our constraint of k/h^2 = 1/3 makes k 
%       sufficiently small so that we can have reasonably accurate
%       results in time. 
%
%   (ii) What is the order of accuracy in Part (II)? Why do you think that
%   you obtain this order of accuracy? 
%
%       In Part (II), we implement a Finite-Difference method with 
%       a 2nd-order approximation with a ghost point x_{m+1}. Since it 
%       is a 2nd-order approximation, we end up with O(h^2) accuracy. 
%       In addition, our constraint of k/h^2 = 1/3 makes k 
%       sufficiently small so that we can have reasonably accurate
%       results in time. 
%
%   (iii) What is the order of accuracy in Part (III)? Why do you think that
%   you obtain this order of accuracy? 
%
%       For Part (III), we implement the Crank-Nicolson method, which 
%       averages the approximations of Forward and Backward Euler 
%       methods, which both have local truncation error O(h^2), resulting
%       in O(h^2 + k^2) = O(h^2) accuracy. 
%       In addition, our constraint of h = k, unlike in Part (I) and
%       Part (II), makes it so that k is not so small, leading to 
%       less accurate approximations. 
%







