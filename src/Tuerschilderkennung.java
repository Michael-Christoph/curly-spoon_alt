import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Created by Michael on 09.08.2017.
 */
public class Tuerschilderkennung {
    public static void main(String[] args){
        Path abs = Paths.get("\\Users\\\\Michael\\\\iss-java-projekt\\\\src\\\\corpus\\\\Krypto-Buch\\\\pic_1.png");
        Path base = Paths.get("\\Users\\Michael\\iss-java-projekt\\src");
        Path rel = base.relativize(abs);
        System.out.println(rel);
    }
}
