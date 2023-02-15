folder = ['result/'];
A = load('iter_size.txt');
x = A(:,1);
y = A(:,2);
z = A(:,3);
f = figure;
plot(y, x,'r-o', 'MarkerEdgeColor','r', 'MarkerFaceColor', 'r', 'MarkerSize', 4)
title('size x iteration', 'FontSize', 16);
xlabel('size', 'FontSize', 16);
ylabel('iteration', 'FontSize', 16);
grid on;
saveas(f, [folder  'size_iter'], 'png')

g = figure;
plot(y, z,'r-o', 'MarkerEdgeColor','r', 'MarkerFaceColor', 'r', 'MarkerSize', 4)
title('size x time', 'FontSize', 16);
xlabel('size', 'FontSize', 16);
ylabel('time, c', 'FontSize', 16);
grid on;
saveas(g, [folder  'size_time'], 'png')

A = load('Eps_iter.txt');
x = A(:,1);
y = A(:,2);
z = A(:,3);
f = figure;
plot(log10(x), y,'r-o', 'MarkerEdgeColor','r', 'MarkerFaceColor', 'r', 'MarkerSize', 4)
title('Epsilon x iteration, size = 100', 'FontSize', 16);
ylabel('iteration', 'FontSize', 16);
xlabel('log10(Epsilon)', 'FontSize', 16);
grid on;
saveas(f, [folder  'Eps_iter'], 'png')

g = figure;
plot(log10(x), z,'r-o', 'MarkerEdgeColor','r', 'MarkerFaceColor', 'r', 'MarkerSize', 4)
title('Epsilon x time, size = 100', 'FontSize', 16);
xlabel('log10(Epsilon)', 'FontSize', 16);
ylabel('time, c', 'FontSize', 16);
grid on;
saveas(g, [folder  'Eps_time'], 'png')