#####三维虚拟环境基础（Three.js）

"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>元宇宙基础环境</title>
    <style>
        body { margin: 0; overflow: hidden; }
        canvas { display: block; }
    </style>
</head>
<body>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
        // 场景设置
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer();
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(renderer.domElement);

        // 添加光源
        const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
        scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(10, 10, 5);
        scene.add(directionalLight);

        // 创建地面
        const groundGeometry = new THREE.PlaneGeometry(20, 20);
        const groundMaterial = new THREE.MeshStandardMaterial({
            color: 0x90EE90,
            roughness: 0.8,
            metalness: 0.2
        });
        const ground = new THREE.Mesh(groundGeometry, groundMaterial);
        ground.rotation.x = -Math.PI / 2;
        scene.add(ground);

        // 创建简单的建筑物
        function createBuilding(x, z, width, depth, height, color) {
            const geometry = new THREE.BoxGeometry(width, height, depth);
            const material = new THREE.MeshStandardMaterial({ color: color });
            const building = new THREE.Mesh(geometry, material);
            building.position.set(x, height/2, z);
            scene.add(building);
            return building;
        }

        // 创建多个建筑物
        const buildings = [
            createBuilding(-5, -5, 2, 2, 3, 0xFF6B6B),
            createBuilding(0, -3, 3, 3, 4, 0x4ECDC4),
            createBuilding(4, 2, 2.5, 2.5, 5, 0x45B7D1),
            createBuilding(-3, 4, 4, 2, 3.5, 0xFFA07A)
        ];

        // 创建虚拟角色（简单立方体代表）
        const playerGeometry = new THREE.BoxGeometry(0.5, 1.5, 0.5);
        const playerMaterial = new THREE.MeshStandardMaterial({ color: 0x3498DB });
        const player = new THREE.Mesh(playerGeometry, playerMaterial);
        player.position.y = 0.75;
        scene.add(player);

        // 相机位置
        camera.position.set(0, 5, 10);
        camera.lookAt(0, 0, 0);

        // 动画循环
        function animate() {
            requestAnimationFrame(animate);

            // 简单的角色移动（可通过键盘事件增强）
            player.rotation.y += 0.01;

            renderer.render(scene, camera);
        }
        animate();

        // 窗口大小调整
        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });

        // 简单的键盘控制
        const moveSpeed = 0.1;
        window.addEventListener('keydown', (event) => {
            switch(event.key) {
                case 'w': // 前移
                    player.position.z -= moveSpeed;
                    break;
                case 's': // 后移
                    player.position.z += moveSpeed;
                    break;
                case 'a': // 左移
                    player.position.x -= moveSpeed;
                    break;
                case 'd': // 右移
                    player.position.x += moveSpeed;
                    break;
                case ' ': // 跳跃
                    player.position.y += 0.5;
                    setTimeout(() => { player.position.y -= 0.5; }, 300);
                    break;
            }
        });

        console.log("元宇宙基础环境已加载。使用WASD键移动，空格键跳跃。");
    </script>
</body>
</html>"""